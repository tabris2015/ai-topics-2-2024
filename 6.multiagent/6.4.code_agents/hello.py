import random

def main():
    name = input("what is your name: ")
    number = random.randint(1, 10)
    print(f"hello {name}, your number is {number}")

if __name__ == "__main__":
    main()