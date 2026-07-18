import random
def hangman():
    print("Welcome to hangman!")
    word_bank = ["apple", "grape", "mango", "peach", "berry"]
    def displayfunction(userguessletters, secretword):
        display = ""
        for letter in secretword:
            if letter in userguessletters:
                display += letter
            else:
                display += "_"
        return(display)
    def guess(secretword):
        
        userguessletters.append(userguess)
        return (displayfunction(userguess,secretword))
    gameover = False
    while True:
        while not gameover:
            random_number=random.randint(0, len(word_bank)-1)   
            secretword=word_bank[random_number]
            
            userguessletters = [] #list of user guesses
            
            if len(userguessletters) >6: #checks if user guessed more than 6 times
                print("You lose!")
                gameover = True
            else:
                userguess = input("Please guess a letter: ")
                print(guess(userguess,secretword))
            if userguess in secretword:
                print("Correct!")
            else:
                print("Wrong Guess!")
                userguess = input("Please guess a letter: ")
                print(guess(userguess,secretword))
            if all(_ in userguessletters for _ in secretword):
                print("You win!!")
                gameover = True
                    
    
        if gameover == True:
            play_again = input("\nDo you want to play again? (yes/no): ")
            if play_again == "yes":
                hangman()  
            elif play_again == "no":
                print ("Game Ended")
            else:
                print ("Invalid input. Play again? (yes/no)") #makes sure that the player enters yes or no


if __name__ == "__main__":
    hangman()

    

