import subprocess
import time
import os
import sys

# Function to help run the command and check for errors
def run_step(command, step_name):
    print(f"⏳ {step_name}...")
    try:
        # Use sys.executable to call the current Python executable
        cmd_list = command.split()
        if cmd_list[0] == 'python':
            cmd_list[0] = sys.executable
            
        subprocess.run(cmd_list, check=True, shell=False)
        print(f"✅ {step_name} Completed!\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {step_name}: {e}")
        sys.exit(1)

def main():
    print("=================================================")
    print("🚀 STARTING HR SYSTEM (DATABASE MODE)")
    print("=================================================\n")

    # --- STEP 1: CHECK DATABASE ---
    if os.path.exists('hr_database.db'):
        print("✅ Found 'hr_database.db'. Using existing database.\n")
    else:
        print("❌ Error: 'hr_database.db' not found!")
        print("   Please place your .db file in this folder.")
        sys.exit(1)

    # --- STEP 2: RUN PIPELINE (Update Data & Model) ---
    # ขั้นตอนนี้จะดึงข้อมูลจาก db ของคุณมาสร้างไฟล์ dashboard_data.csv
    run_step("python auto_pipeline.py", "Processing Data from Database")

    # --- STEP 3: LAUNCH DASHBOARD ---
    print("📊 Launching Dashboard...")
    print("=================================================")
    print("🌐 Opening in your browser... Press Ctrl+C to stop.")
    
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 System Stopped.")

if __name__ == "__main__":
    main()