import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_model import train_model
from detect_anxiety import detect_anxiety

def main():
    print("Anxiety Detection Project")
    print("1. Train the model")
    print("2. Run anxiety detection")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        train_model()
    elif choice == '2':
        detect_anxiety()
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()