import argparse
from encryption import Encryptor
from recognize import Recognizer
from otp_verification import send_otp_email, generate_otp, verify_otp, validate_email
from password_verification import verify_password, verify_username
import time
import threading

# User-label mapping
user_mapping = {
    "Samyukta": "user0",
    "Aryan":"user1",
    "Shivangi": "user2",
    "Vinithra": "user3",
    # Add more users as needed
}

# User-email mapping
user_email_mapping = {
    "Samyukta": "samyukta.pathak2021@vitstudent.ac.in",
    "Aryan": "aryan@example.com",
    "Shivangi": "shivangi@example.com",
    "Vinithra": "vinithra@example.com",
    # Add more users as needed
}

# Function to authenticate user based on password
def authenticate_user_password():
    global username
    attempts = 3
    while attempts > 0:
        username = input("Enter Your Username: ")
        if verify_username(username):
            password_attempts = 3
            while password_attempts > 0:
                password = input("Enter Your Password: ")
                if verify_password(username, password):
                    print("Password Verified.")
                    print()
                    return username  # Return the username upon successful authentication
                password_attempts -= 1
            print("Too Many Incorrect Attempts.")
            print("Please try again later.")
            return None
        attempts -= 1
    print("Authentication Failed. Too Many Attempts.")
    return None

def update_time_left(start_time, event, time_limit):
    while not event.is_set():
        time_passed = time.time() - start_time
        time_left = max(0, time_limit - time_passed)  # Calculate time left
        if time_left == 0:
            break
        print(f"\rTime Left: {time_left} seconds", end='', flush=True)
        time.sleep(1)

def authenticate_user_otp(username):
    attempts = 3
    receiver_email = user_email_mapping.get(username)
    if not validate_email(receiver_email):
        print("Your Registered Email is Invalid.")
        print("Contact Admin to Update.")
        exit()

    generated_otp = generate_otp()
    send_otp_email(receiver_email, generated_otp)
    print("OTP has been sent to your Registered Email.")

    start_time = time.time()
    time_limit = 300  # Time limit for OTP verification

    while attempts > 0:
        timer_event = threading.Event()
        time_thread = threading.Thread(target=update_time_left, args=(start_time, timer_event, time_limit))
        time_thread.start()
        print("\n")
        timer_event.set()

        input_otp = input("Enter the OTP received: ")

        time_thread.join()

        if verify_otp(input_otp, generated_otp):
            print("\nOTP Verified.")
            print()
            print("Please Look at the Camera.")
            return True
        else:
            print("\nInvalid OTP.")
            print("Please Try Again.")
            print()
        attempts -= 1

    print("OTP Verification Failed.")
    return False

# Function to authenticate user based on detected faces
def authenticate_user_face():
    recog = Recognizer(user_mapping)
    user_name = recog.recognize()
    return user_name

# Main functionality
def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="Path to file")
    ap.add_argument("-m", "--mode", required=True, help="Enter 'encrypt' or 'decrypt'")
    args = vars(ap.parse_args())

    # Extract arguments
    file_path = args['file']
    mode = args['mode']

    # Encryption key
    key = b'[EX\xc8\xd5\xbfI{\xa2$\x05(\xd5\x18\xbf\xc0\x85)\x10nc\x94\x02)j\xdf\xcb\xc4\x94\x9d(\x9e'
    enc = Encryptor(key)

    # Main functionality
    if mode == "encrypt":
        try:
            enc.encrypt_file(file_path)
            print("File Encrypted")
        except:
            print("Incorrect File Path")
        exit()

    elif mode == "decrypt":
                                  
        # Authenticate user using password
        if not authenticate_user_password():
            exit()
            
        # Authenticate user using otp
        if not authenticate_user_otp(username):
            exit()

        # Authenticate user using face detection
        authenticated_user = authenticate_user_face()
        if authenticated_user:
            try:
                enc.decrypt_file(file_path)
                print()
                print("File Decrypted.")
                print(f"User Authenticated: {authenticated_user}")
            except Exception as e:
                print("Decryption Failed:", e)
        else:
            print("Face Authentication Failed. Exiting.")

    else:
        print("Incorrect Mode")

if __name__ == "__main__":
    main()
