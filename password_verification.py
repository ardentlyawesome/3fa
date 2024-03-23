user_password_mapping = {
        "Samyukta": "password0",
        "Aryan": "password1",
        "Shivangi": "password3",
        "Vinithra": "password4",
        # Add more users as needed
    }

def verify_username(username):
    # Verify if the entered username is in the mapping
    if username in user_password_mapping:
        return True
    else:
        print("User Not Registered.")
        print("Please Try Again.")
        print()
        return False
    
def verify_password(username, password):
    # Verify password for the given username
    if user_password_mapping.get(username) == password:
        return True
    else:
        print("Incorrect Password.")
        print("Please Try Again.")
        print()
        return False
