from tkinter import *
import pygame
from tkinter import filedialog
from extract import *
import test_1
from test_1 import *
# from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter.filedialog import askopenfile
import os.path
import random
import string

root = Tk()
root.geometry("600x150")
root.title("Sound_GUI")
# root.call('wm', 'iconphoto',Tk._w,ImageTk.PhotoImage(Image.open('encrypo.ico')))
root.configure(background="#D9D8D7")
pygame.mixer.init()


def hide_frame():
    first_frame.grid_forget()

def extract1():
    mask_list = [("Sound files", "*.wav")]
    lang = extract(filedialog.askopenfilename(initialdir='', filetypes=mask_list))
    messagebox.showinfo("Feature Extract", lang)
    
def Feature():
    mask_list = [("Sound files", "*.csv")]
    lang = gui(filedialog.askopenfilename(initialdir='', filetypes=mask_list))
    if (lang == 0):
        Y1 = 'asm'

    elif (lang == 1 ):
        Y1 = 'ben'

    elif (lang == 2):
        Y1 = 'guj'

    elif (lang == 3):
        Y1 = 'hin'

    elif (lang == 4):
        Y1 = 'kan'

    elif (lang == 5):
        Y1 = 'mal'

    elif (lang == 6 ):
        Y1 = 'odi'

    elif (lang == 7):
        Y1 = 'tel'
    messagebox.showinfo("Feature", Y1)


def play_sound():
    mask_list = [("Sound files", "*.wav *.au *.mp3")]
    sound_file = filedialog.askopenfilename(initialdir='', filetypes=mask_list)

    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()


# Add a Button widget
def stop_sound():
    pygame.mixer.music.stop()



def file_one():
    hide_frame()
    first_frame.grid(sticky=N + S + W + E)
    first_frame.configure(background="#D9D8D7")
    

    fileButton = Button(first_frame, text="extract", fg="white", bg="OrangeRed4", command=extract1,
                        activeforeground="black", activebackground="coral", relief="raised", bd=8)
    fileButton.grid(row=0, column=1, padx=20, pady=30)

    button_quit = Button(first_frame, text="EXIT", command=root.quit, fg="white", bg="OrangeRed4",
                         activeforeground="black", activebackground="coral", relief="raised", bd=9)
    button_quit.grid(row=0, column=2, padx=20, pady=20)

    b1 = Button(first_frame, text="Play Music", command=play_sound, fg="white", bg="OrangeRed4",
                activeforeground="black", activebackground="coral", relief="raised", bd=10)
    b1.grid(row=0, column=3, padx=20, pady=20)

    b2 = Button(first_frame, text="Stop Music", command=stop_sound, fg="white", bg="OrangeRed4",
                activeforeground="black", activebackground="coral", relief="raised", bd=10)
    b2.grid(row=0, column=4, padx=20, pady=20)

    b3 = Button(first_frame, text="Feature", command=Feature, fg="white", bg="OrangeRed4",
                activeforeground="black", activebackground="coral", relief="raised", bd=10)
    b3.grid(row=0, column=5, padx=20, pady=20)


menubar = Menu(root)
menubar.add_command(label="ENCRYPT", activebackground="OrangeRed4", activeforeground="black", command=file_one)
# menubar.add_command(label="DECRYPT",activebackground="OrangeRed4",activeforeground="black",command=file_two)
# menubar.add_command(label="ABOUT",activebackground="OrangeRed4",activeforeground="black",command=file_three)
root.config(menu=menubar)
first_frame = Frame(root, width=600, height=400)
file_one()
root.mainloop()
