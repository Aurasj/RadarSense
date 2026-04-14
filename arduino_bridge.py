import socket
import serial
import sys
import time
import datetime
import threading

# ==========================================
# CONFIGURARE
# ==========================================
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
COM_PORT = "COM6"  # Asigură-te că portul este corect
BAUD_RATE = 9600

# WATCHDOG TIMER (în secunde)
# Cum `serverV2.py` știe acum să trimită "none" instant când iei mâna, acest Watchdog a devenit 
# doar o plasă de siguranță pentru gesturile scurte (ex: Tap, Push). L-am coborât la 0.2s!
WATCHDOG_TIMEOUT = 0.2  

COMMAND_MAP = {
    "hold": "F",   # Înainte
    "tap":  "B",   # Înapoi
    "push": "L",   # Stânga curba pt scurt timp
    "pull": "R",   # Dreapta curba pt scurt timp
    "wave": "M",   # Wave = Cântă
}

# ==========================================
# PORNIRE CONEXIUNI
# ==========================================
print(f"🔄 Conectare la mașina Arduino pe {COM_PORT}...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1, dsrdtr=True)
    time.sleep(1.5)  # Timp de inițializare modul Bluetooth
    print("✅ CONECTAT! Sistemul de frânare cu Watchdog activat.")
except Exception as e:
    print(f"❌ EROARE CONEXIUNE SERIALĂ: {e}")
    ser = None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.5) # Time-out pentru a debloca ascultarea și a permite Ctrl+C

try:
    sock.bind((UDP_IP, UDP_PORT))
except OSError as e:
    print(f"❌ Eroare binding port UDP {UDP_PORT}: {e}")
    sys.exit(1)

last_cmd = "S"
last_udp_time = time.time()

def send_cmd(cmd):
    global last_cmd
    # Trimitem doar la SCHIMBARE, nu de fiecare dată
    if cmd != last_cmd:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🏎️  Transmis Bluetooth: [{cmd}]")
        if ser:
            try:
                ser.write(cmd.encode())
            except Exception:
                pass
        last_cmd = cmd

# ==========================================
# THREAD FRÂNĂ AUTOMATĂ (WATCHDOG TIMER)
# ==========================================
def watchdog():
    """
    Oprește mașina instaneu dacă serverul tace.
    Asta rezolvă problema cu 'luatul mâinii'. UDP-ul dă "Hold", iar 
    dacă nu mai primește alt "Hold" în 0.5s (că ai luat mâna), frânează.
    """
    while True:
        if time.time() - last_udp_time > WATCHDOG_TIMEOUT:
            send_cmd("S")
        time.sleep(0.05)

t = threading.Thread(target=watchdog, daemon=True)
t.start()

# ==========================================
# BUCLA PRINCIPALĂ (ASCULTARE UDP)
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 Bridge-ul este activ pe UDP port {UDP_PORT} (Identic cu Unity).")
    print("💡 Apasă Ctrl+C în orice moment pentru a opri mașina instant.\n")

    try:
        while True:
            # Datorită timeout-ului de la socket.recvfrom, Windows ne va da voie să dăm Ctrl+C o dată la 0.5s!
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                continue

            gesture = data.decode().strip().lower()
            
            # Un gest nou a fost confirmat! (Resetăm cronometrul / ținem mașina 'trează')
            last_udp_time = time.time()

            cmd = COMMAND_MAP.get(gesture, "S")
            
            # Executăm DIRECT orice comandă validă, inclusiv "S" (care vine instant ca "none" de la radar)
            if cmd == "S":
                if gesture == "none":
                    print(f"👇 Radar GOL -> Frână EXTREMĂ 0ms!")
            else:
                print(f"📡 UnityUDP Primit: {gesture.upper():<5}")
                
            send_cmd(cmd)

    except KeyboardInterrupt:
        print("\n🛑 S-a apăsat Ctrl+C! Oprire de urgență!")
        send_cmd("S")
        send_cmd("N")  # Oprim și muzica 
        if ser:
            ser.close()
            print("✅ Port serial UART închis în siguranță.")
        sock.close()
        print("✅ Ieșire curată.")
        sys.exit(0)