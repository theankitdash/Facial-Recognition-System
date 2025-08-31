import cv2, time, random, numpy as np, torch, os, mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1

# ================= FACE RECOGNITION SETUP ==================
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

USER_DIR = "faces_db"
os.makedirs(USER_DIR, exist_ok=True)

# ================= MEDIAPIPE SETUP ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
NOSE_TIP = 1
MOUTH = [61, 291, 0]  # left, right, center (approx)

# ================== LIVENESS CHECKS ==================
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def blink_check(timeout=15):
    cap = cv2.VideoCapture(0)
    print("👉 Challenge: Please BLINK...")
    blinked, start_time = False, time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear_left = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            ear_right = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear = (ear_left + ear_right) / 2.0
            if ear < 0.22: blinked = True

        cv2.putText(frame, "Blink Detected!" if blinked else "Please BLINK",
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Liveness - Blink", frame)

        if blinked or time.time() - start_time > timeout: break
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()
    return blinked

def head_movement_check(timeout=15):
    cap = cv2.VideoCapture(0)
    print("👉 Challenge: Please TURN your head LEFT/RIGHT...")
    moved, ref_x, start_time = False, None, time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            nose_x = int(lm[NOSE_TIP].x * w)
            if ref_x is None: ref_x = nose_x
            else:
                if abs(nose_x - ref_x) > 40: moved = True

        cv2.putText(frame, "Head Movement Detected!" if moved else "Please TURN head",
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Liveness - Head", frame)

        if moved or time.time() - start_time > timeout: break
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()
    return moved

def smile_check(timeout=15):
    cap = cv2.VideoCapture(0)
    print("👉 Challenge: Please SMILE...")
    smiled, start_time = False, time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left = np.array([lm[MOUTH[0]].x * w, lm[MOUTH[0]].y * h])
            right = np.array([lm[MOUTH[1]].x * w, lm[MOUTH[1]].y * h])
            center = np.array([lm[MOUTH[2]].x * w, lm[MOUTH[2]].y * h])
            mouth_width = np.linalg.norm(left - right)
            mouth_height = np.linalg.norm(center - ((left+right)/2))
            if mouth_height / mouth_width > 0.3: smiled = True

        cv2.putText(frame, "Smile Detected!" if smiled else "Please SMILE",
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Liveness - Smile", frame)

        if smiled or time.time() - start_time > timeout: break
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()
    return smiled

def liveness_check():
    challenges = [blink_check, head_movement_check, smile_check]
    random.shuffle(challenges)
    for check in challenges:
        if not check():
            print("❌ Liveness failed. Access denied.")
            return False
    return True

# ================== FACE RECOGNITION ==================
def get_embedding(img_bgr):
    face = mtcnn(img_bgr)
    if face is None: return None
    with torch.no_grad():
        return facenet(face.unsqueeze(0)).cpu().numpy()

def register_face(username):
    cap = cv2.VideoCapture(0)
    print(f"Registering {username}...")
    samples = []
    while len(samples) < 50:
        ret, frame = cap.read()
        if not ret: continue
        emb = get_embedding(frame)
        if emb is not None:
            samples.append(emb)
            print(f"Captured {len(samples)}/50")
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()
    if samples:
        mean_emb = np.mean(np.vstack(samples), axis=0, keepdims=True)
        np.save(os.path.join(USER_DIR, f"{username}.npy"), mean_emb)
        print(f"✅ {username} registered.")
        cv2.putText(frame, f"{username} Registered!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Success", frame); cv2.waitKey(2000); cv2.destroyAllWindows()

def recognize_face():
    if not liveness_check():
        print("❌ Liveness failed. Access denied.")
        return

    db = {f[:-4]: np.load(os.path.join(USER_DIR,f))
          for f in os.listdir(USER_DIR) if f.endswith(".npy")}
    if not db:
        print("No users registered."); return

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release(); cv2.destroyAllWindows()

    emb = get_embedding(frame)
    if emb is None:
        print("No face detected."); return

    def cos_sim(a,b): return float(np.dot(a,b.T)/(np.linalg.norm(a)*np.linalg.norm(b)))
    best_user, best_score = None, -1
    for user, ref_emb in db.items():
        score = cos_sim(emb, ref_emb)
        if score > best_score:
            best_user, best_score = user, score

    if best_score > 0.75:
        print(f"✅ Recognized as {best_user} (similarity={best_score:.3f})")
        cv2.putText(frame, f"Welcome {best_user}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        print("❌ Face not recognized.")
        cv2.putText(frame, "Access Denied", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Result", frame); cv2.waitKey(3000); cv2.destroyAllWindows()

# ================== MAIN MENU ==================
def main_menu():
    cap = cv2.VideoCapture(0)
    print("👉 Press 'r' to Register, 'l' to Login, 'q' to Quit")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Draw buttons
        cv2.rectangle(frame, (50,50), (250,120), (0,255,0), -1)
        cv2.putText(frame, "Register (r)", (60,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.rectangle(frame, (300,50), (500,120), (255,255,0), -1)
        cv2.putText(frame, "Login (l)", (310,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.rectangle(frame, (550,50), (750,120), (0,0,255), -1)
        cv2.putText(frame, "Quit (q)", (560,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("FaceAuth Menu", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            cap.release(); cv2.destroyAllWindows()
            name = input("Enter your name: ")
            register_face(name)
            cap = cv2.VideoCapture(0)  # reopen camera
        elif key == ord("l"):
            cap.release(); cv2.destroyAllWindows()
            recognize_face()
            cap = cv2.VideoCapture(0)  # reopen camera
        elif key == ord("q"):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main_menu()
