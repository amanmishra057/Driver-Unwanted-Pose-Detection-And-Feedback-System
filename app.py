from flask import Flask, render_template, request, redirect, url_for, session, Response, abort
from flask_sqlalchemy import SQLAlchemy
import bcrypt   
import cv2
from playsound import playsound
from threading import Thread, Event
from datetime import datetime
import os
from pose_detection import detect_pose
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Global event for controlling sound
sound_stop_event = Event()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'processed')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

db = SQLAlchemy(app)
app.secret_key = 'secret_key'

# Ensure upload folders exist



if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

    def __init__(self, name, username, email, password):
        self.name = name
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        
    def check_password(self, password):
        return bcrypt.checkpw(password.encode("utf-8"), self.password.encode("utf-8"))

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(100), nullable=False)
    user_email = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, alert_type, user_email):
        self.alert_type = alert_type
        self.user_email = user_email

class ScreenshotAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    user_email = db.Column(db.String(120), nullable=False)
    alert_type = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()


def play_alert_sound():
    try:
        # Reset the event flag
        sound_stop_event.clear()
        while not sound_stop_event.is_set():
            playsound("static/resources/beep.mp3")
            # Small delay to prevent CPU overload
            sound_stop_event.wait(0.1)
    except Exception as e:
        print("Sound alert error:", e)

def process_frame(frame, user_email):
    try:
        is_unwanted, class_name, confidence = detect_pose(frame)
        label = f"{class_name}: {confidence * 100:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  
                    (0, 0, 255) if is_unwanted else (0, 255, 0), 2)
        return frame, is_unwanted, class_name, confidence
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, False, "Error", 0

def gen_frames(user_email):
    if user_email.endswith("@poseguard.com"):
        return

    unwanted_pose_count = 0
    threshold = 10
    last_alert_time = datetime.now()
    cooldown_seconds = 10
    sound_thread = None
    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Failed to grab frame")
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:  # Skip frames to reduce processing load
                # Still display frame but skip processing
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            # Process frame
            is_unwanted, class_name, confidence = detect_pose(frame)
            
            # Draw prediction results
            label = f"{class_name}: {confidence * 100:.1f}%"
            color = (0, 0, 255) if is_unwanted else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if is_unwanted:
                unwanted_pose_count += 1
            else:
                unwanted_pose_count = 0
                if sound_thread and sound_thread.is_alive():
                    sound_stop_event.set()

            current_time = datetime.now()
            time_since_last_alert = (current_time - last_alert_time).total_seconds()

            if unwanted_pose_count > threshold and time_since_last_alert >= cooldown_seconds:
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                filename = f"{user_email}_{timestamp}.jpg"
                image_path = f"static/screenshots/{filename}"
                cv2.imwrite(image_path, frame)

                with app.app_context():
                    new_alert = Alert(alert_type=f"Unwanted Pose Detected ({class_name})", user_email=user_email)
                    db.session.add(new_alert)

                    screenshot_alert = ScreenshotAlert(
                        image_path=image_path,
                        user_email=user_email,
                        alert_type=f"Unwanted Pose Detected ({class_name})"
                    )
                    db.session.add(screenshot_alert)
                    db.session.commit()

                if not sound_thread or not sound_thread.is_alive():
                    sound_thread = Thread(target=play_alert_sound)
                    sound_thread.start()
                
                last_alert_time = current_time
                unwanted_pose_count = 0

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Reduce JPEG quality for faster transmission
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in frame processing: {e}")
            continue


@app.route('/video-feed/<filename>')
def video_feed_with_filename(filename):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    def generate_frames():
        video_path = os.path.join(tempfile.gettempdir(), secure_filename(filename))
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return
            
        frame_skip = 2  # Process every 2nd frame to improve performance
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                    
                try:
                    is_unwanted, class_name, confidence = detect_pose(frame)
                    label = f"{class_name}: {confidence * 100:.1f}%"
                    color = (0, 0, 255) if is_unwanted else (0, 255, 0)
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
        except Exception as e:
            print(f"Error in generate_frames: {e}")
        finally:
            cap.release()
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    if session.get('admin'):
        abort(403)  # Admins don't have access to video feed
    return Response(gen_frames(session.get('user')), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if not all([username, email, password]):
            error = 'All fields are required'
        # elif password != confirm_password:
        #     error = 'Passwords do not match'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters long'
        elif not any(char.isdigit() for char in password):
            error = 'Password must contain at least one digit'
        elif not any(char.isalpha() for char in password):
            error = 'Password must contain at least one letter'
        elif not any(char in '!@#$%^&*()_+' for char in password):
            error = 'Password must contain at least one special character'
        elif not any(char.isupper() for char in password):
            error = 'Password must contain at least one uppercase letter'
        else:
            user = User.query.filter((User.username == username) | (User.email == email)).first()
            if user:
                error="Username or Email already exists!"
            new_user = User(name, username, email, password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))

    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_username = request.form['username']
        password = request.form['password']
        user = User.query.filter((User.username == email_or_username) | (User.email == email_or_username)).first()

        if user and user.check_password(password):
            session['user'] = user.email
            if user.email.endswith("@poseguard.com"):
                session['admin'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

# In app.py
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    user_email = session.get('user')
    is_admin = user_email.endswith('@poseguard.com')
    return render_template('dashboard.html', is_admin=is_admin)


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('admin', None)
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/information')
def information():
    return render_template('information.html')


@app.route('/admin')
def admin_portal():
    if 'user' not in session or not session['user'].endswith('@poseguard.com'):
        abort(403)
    
    # Fetch screenshot alerts
    shots = ScreenshotAlert.query.order_by(ScreenshotAlert.timestamp.desc()).all()
    
    return render_template('admin_portal.html', shots=shots)


@app.route('/reports')
def reports():
    if 'user' not in session or not session['user'].endswith('@poseguard.com'):
        abort(403)
    
    # Fetch alerts and screenshots
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    screenshots = ScreenshotAlert.query.order_by(ScreenshotAlert.timestamp.desc()).all()
    
    return render_template('reports.html', alerts=alerts, screenshots=screenshots)

@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403


@app.route('/video-upload')
def video_upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('video_upload.html')

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if 'video' not in request.files:
        return 'No video file uploaded', 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected file', 400
    
    if not allowed_file(video_file.filename):
        return 'Invalid file type', 400

    # Save uploaded file temporarily
    filename = secure_filename(video_file.filename)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    video_file.save(temp_path)
    
    # Return the template with the filename for video processing
    return render_template('video_upload.html', filename=filename)


if __name__ == '__main__':
    if not os.path.exists('static/screenshots'):
        os.makedirs('static/screenshots')
    app.run(debug=True)