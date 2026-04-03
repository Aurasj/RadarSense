import socketio
import socket
import time

# --- CONFIGURARE ---
SERVER_URL = "http://localhost:5002"
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5005

# Timpi de pauză personalizați (în secunde) pentru fiecare gest
COOLDOWNS = {
    'push': 0.4,  # Rapid, pentru a accelera/urca cursiv
    'pull': 0.4,  # Rapid, pentru a frâna/coborî cursiv
    'hold': 0.5,
    'wave': 1.0,  # Gest tactic (meniu/schimbare tactică) - pauză mare
    'tap':  1.0,  # Toggle Autopilot - pauză mare ca să nu facă dublu-click
    'none': 0.1
}

# Pregătire Socket UDP pentru Unity
unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sio = socketio.Client()

last_send_time = 0
last_gesture_sent = "none"

@sio.event
def connect():
    print("[BRIDGE] Conectat la serverul AI! Gata pentru SWARM CONTROL.")

@sio.on('radar_data')
def on_data(data):
    global last_send_time, last_gesture_sent
    
    gesture = data.get('gesture_out', 'none')
    now = time.time()

    # 1. Ignorăm datele cât timp se calibrează
    if gesture == 'baseline':
        return

    # 2. Logică inteligentă de trimitere
    if gesture != 'none':
        is_new_gesture = (gesture != last_gesture_sent)
        cooldown_passed = (now - last_send_time) > COOLDOWNS.get(gesture, 0.5)

        # Trimitem dacă e un gest nou SAU dacă am ținut mâna suficient timp (cooldown)
        if is_new_gesture or cooldown_passed:
            unity_sock.sendto(gesture.encode(), (UNITY_IP, UNITY_PORT))
            print(f"[{time.strftime('%H:%M:%S')}] 🚀 COMANDĂ TACTICĂ: {gesture.upper()}")
            
            last_send_time = now
            last_gesture_sent = gesture
            
    else:
        # Dacă luăm mâna de pe senzor ('none')
        if last_gesture_sent != 'none':
            unity_sock.sendto(b"none", (UNITY_IP, UNITY_PORT))
            print(f"[{time.strftime('%H:%M:%S')}] 🛑 Mână retrasă (Idle)")
            last_gesture_sent = 'none'

@sio.on('status')
def on_status(data):
    if not data.get('baseline_ready'):
        print("[BRIDGE] ⏳ Se calibrează senzorul (Nu mișca mâna)...")

if __name__ == '__main__':
    print("="*50)
    print("  LINK DE DATE UDP (UNITY BRIDGE) - ACTIVAT")
    print("="*50)
    try:
        sio.connect(SERVER_URL)
        sio.wait()
    except Exception as e:
        print(f"[EROARE] Conexiune eșuată: {e}. Asigură-te că server.py rulează!")