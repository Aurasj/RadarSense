import socketio
import socket
import time

# --- CONFIGURARE ---
SERVER_URL = "http://localhost:5002"
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5005
COOLDOWN_TIME = 0.5  # Secunde de pauză între două comenzi trimise

# Pregătire Socket UDP pentru Unity
unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sio = socketio.Client()

last_send_time = 0
last_gesture_sent = "none"

@sio.event
def connect():
    print("[BRIDGE] Conectat la serverul web radar!")

@sio.on('radar_data')
def on_data(data):
    global last_send_time, last_gesture_sent
    
    gesture = data.get('gesture_out', 'none')
    now = time.time()

    # LOGICĂ: 
    # 1. Să nu fie 'none' sau 'baseline'
    # 2. Să fi trecut măcar 0.5 secunde de la ultima trimitere
    # 3. SAU să fie un gest diferit de ultimul trimis
    if gesture not in ['none', 'baseline']:
        if (now - last_send_time > COOLDOWN_TIME) or (gesture != last_gesture_sent):
            unity_sock.sendto(gesture.encode(), (UNITY_IP, UNITY_PORT))
            print(f"[BRIDGE] Gest trimis: {gesture}")
            last_send_time = now
            last_gesture_sent = gesture

@sio.on('status')
def on_status(data):
    if not data.get('baseline_ready'):
        print("[BRIDGE] Serverul se calibrează (Wait for baseline)...")

if __name__ == '__main__':
    try:
        sio.connect(SERVER_URL)
        sio.wait()
    except Exception as e:
        print(f"[EROARE] Conexiune eșuată: {e}")