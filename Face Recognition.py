import csv, time, datetime, cv2, os, shutil, glob, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk #treeview
from PIL import ImageTk, Image
from tkinter import messagebox

root=Tk()
root.geometry("1600x800")
root.title("ATTENDANCE SYSTEM")
root.configure(background="snow")

WIDTH, HEIGHT = 1600,800
root.geometry("{}x{}".format(WIDTH, HEIGHT))

background_image = ImageTk.PhotoImage(Image.open("background.jpg").resize((WIDTH, HEIGHT), Image.ANTIALIAS))

title_font = ('Times New Roman', 35, 'bold')
button_font = ('Times New Roman', 15, ' bold ')

fgColor = 'white'
bgColor = 'blue'

def track_images():
    tv = ttk.Treeview(attendance_frame,height =13,columns = ('name','date','time'))
    tv.column('#0',width=82)
    tv.column('name',width=130)
    tv.column('date',width=133)
    tv.column('time',width=133)
    tv.place(x=565, y=300)
    tv.heading('#0',text ='ID')
    tv.heading('name',text ='NAME')
    tv.heading('date',text ='DATE')
    tv.heading('time',text ='TIME')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer\\trainer.yml')
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    #VARIABEL
    font = cv2.FONT_HERSHEY_SIMPLEX

    # video capture
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    # read file
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    attendance = pd.DataFrame(columns = col_names)
    df=pd.read_csv("user\\user.csv")
    # in order to stop within a certain time
    frame_step = 7
    for frame in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame * frame_step)
        # read frame
        ret, frame = cap.read()
        # Display
        frame = cv2.flip(frame, 1) # Flip vertical
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect face
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
        )
        i = 0
        for (x, y, w, h) in faces:            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence < 100):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['face_id'] == Id]['face_name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,'',aa,'',date,'',timeStamp]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                Id = "unknown"
                tt=str(Id)
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(frame, str(tt), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        
        cv2.imshow('img',frame)

        k = cv2.waitKey(100) & 0xff 
        if k == 27:
            break

    # close program
    cap.release()
    cv2.destroyAllWindows()

    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    attendance=attendance.drop_duplicates(subset=['Id', 'Date'],keep='first')
    fileName="attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    print(attendance)
    with open("attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                iidd = str(lines[0])
                tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    csvFile1.close()
    plt.show()
    
def instruction():
    msg = messagebox.showinfo('instruction','1. Select Menu (Face Registration) for New Face Registration\n2. Select the (Attendance) Menu to do Attendance\n3. Select Menu (Exit) to Exit Attendance System')

take_image_frame=None
def take_image():
    global take_image_frame
    take_image_frame = Frame(home)
    take_image_frame.place(x=0,y=0)
    take_image_frame.tkraise()
    take_image_title = Label(take_image_frame, text="NEW FACE REGISTRATION", font=title_font).pack()
    take_image_bg=Label(take_image_frame,image=background_image).pack()
    back_button1 = Button(home, text='Home',command=main_window,fg=fgColor,bg=bgColor,width=15,height=1, activebackground = "snow" ,font=('Times New Roman', 15, ' bold '))
    back_button1.place(x=710, y=650)

    def delete1():
        txt.delete(0, 'end')
        res = "1. Take Image - 2. Training"
        message1.configure(text=res)

    def delete2():
        txt2.delete(0, 'end')
        res = "1. Take Image - 2. Training"
        message1.configure(text=res)
    
    def take_img():
        face_id = (txt.get())
        face_name = (txt2.get())
        if ((face_name.isalpha()) or (' ' in face_name)):
            cam = cv2.VideoCapture(0)
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            count = 0
            while(True):
                ret, img = cam.read(0)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                    roi = img[y:y + h, x:x + w]

                    # Save the captured image into the datasets folder
                    if count %2 == 0:
                        cv2.imwrite("dataset/"+ str(face_name) +'.'+ str(face_id) + '.' + str(count) + ".jpg", roi)
                    count += 1

                    cv2.imshow('image', img)

                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 120: # Take 60 face sample and stop video
                    break

            cam.release()
            cv2.destroyAllWindows()
            res = "Image captured by ID : " + face_id
            row = [face_id , face_name]
            with open('user\\user.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            message1.configure(text=res)
        else:
            if (face_name.isalpha() == False):
                res = "Please Enter the Name Correctly"
                message.configure(text=res)
    def TrainImages():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces, ID = getImagesAndLabels("dataset")
        try:
            recognizer.train(faces, np.array(ID))
        except:
            messagebox._show(title='No registration', message='Please register first')
            return
        recognizer.save("trainer\\trainer.yml")
        res = "Profile saved successfully"
        message1.configure(text=res)
        message.configure(text='Total Registration : ' + str(ID[0]))

    ############################################################################################3

    def getImagesAndLabels(path):
        # get path dari semua file dalam folder
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # membuat face list kosong
        faces = []
        # membuat id list kosong
        Ids = []
        # looping semua gambar dalam path dan loading id, image
        for imagePath in imagePaths:
            # loading gambar dan converting ke gray scale
            pilImage = Image.open(imagePath).convert('L')
            # converting PIL image ke numpy array
            imageNp = np.array(pilImage, 'uint8')
            # mendapatkan id dari gambar
            ID = int(os.path.split(imagePath)[-1].split(".")[1])
            # ekstrak face dari sampel training image
            faces.append(imageNp)
            Ids.append(ID)
        return faces, Ids

    lbl = Label(take_image_frame, text="Enter ID",width=35  ,height=1  ,fg=fgColor  ,bg="black" ,font=('times',  16, ' bold ') )
    lbl.place(x=593, y=175)

    txt = Entry(take_image_frame,width=32 ,fg="black",font=('times', 15, ' bold '))
    txt.place(x=593, y=208)

    lbl2 = Label(take_image_frame, text="Enter Name",width=35 ,height=1 ,fg=fgColor  ,bg="black" ,font=('times', 16, ' bold '))
    lbl2.place(x=593, y=260)

    txt2 = Entry(take_image_frame,width=32 ,fg="black",font=('times', 15, ' bold '))
    txt2.place(x=593, y=293)

    message1 = Label(take_image_frame, text="1. Take Image - 2. Training" ,fg=fgColor  ,bg="black"  ,width=35 ,height=1, activebackground = "yellow" ,font=('times', 15, ' bold '))
    message1.place(x=593, y=340)

    message = Label(take_image_frame, text="" ,fg=fgColor  ,bg="black"  ,width=35,height=1, activebackground = "yellow" ,font=('times', 16, ' bold '))
    message.place(x=593, y=500)

    tombol_hapus1 = Button(take_image_frame, text="Delete", command=delete1  ,fg="black"  ,bg="#ea2a2a"  ,width=11 ,activebackground = "white" ,font=('times', 11, ' bold '))
    tombol_hapus1.place(x=910, y=208)
    tombol_hapus2 = Button(take_image_frame, text="Delete", command=delete2  ,fg="black"  ,bg="#ea2a2a"  ,width=11 , activebackground = "white" ,font=('times', 11, ' bold '))
    tombol_hapus2.place(x=910, y=293)
    take_image = Button(take_image_frame, text="Take Image", command=take_img  ,fg="white"  ,bg="blue"  ,width=35  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    take_image.place(x=593, y=390)
    train_image = Button(take_image_frame, text="Training", command=TrainImages ,fg="white"  ,bg="blue"  ,width=35  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    train_image.place(x=593, y=440)

    res=0
    exists = os.path.isfile("user\\user.csv")
    if exists:
        with open("user\\user.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                res = res + 1
        res = (res // 2) - 1
        csvFile1.close()
    else:
        res = 0
    message.configure(text='Total Registration : '+str(res))

attendance_frame=None
def attendance():
    global attendance_frame
    attendance_frame = Frame(home)
    attendance_frame.place(x=0,y=0)
    attendance_frame.tkraise()
    attendance_title = Label(attendance_frame, text="ATTENDANCE", font=title_font).pack()
    attendance_bg=Label(attendance_frame,image=background_image).pack()
    back_button2 = Button(home, text='Home',command=main_window,fg=fgColor,bg=bgColor,width=15,height=1, activebackground = "snow" ,font=('Times New Roman', 15, ' bold '))
    back_button2.place(x=710, y=650)
    
    track_image = Button(attendance_frame, text="Press to Start Attendance", command=track_images,fg=fgColor,bg=bgColor,width=20,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    track_image.place(x=685, y=175)

    lbl3 = Label(attendance_frame, text="Attendance Data",width=35  ,fg=fgColor,bg="black" ,height=1 ,font=('times', 15, ' bold '))
    lbl3.place(x=595, y=240)
    
    tv = ttk.Treeview(attendance_frame,height =13,columns = ('name','date','time'))
    tv.column('#0',width=82)
    tv.column('name',width=130)
    tv.column('date',width=133)
    tv.column('time',width=133)
    tv.place(x=565, y=300)
    tv.heading('#0',text ='ID')
    tv.heading('name',text ='NAME')
    tv.heading('date',text ='DATE')
    tv.heading('time',text ='TIME')

home=None
def main_window():
    global home
    home = Frame(root, width=WIDTH, height=HEIGHT)
    home.place(x=-120, y=0)
    home_title = Label(home, text="WELCOME TO ATTENDANCE SYSTEM", font=title_font).pack()
    home_bg=Label(home,image=background_image).pack()
    
    instruction_button = Button(home, text='Instruction', command=instruction,fg=fgColor,bg=bgColor,width=15,height=1, activebackground = "snow" ,font=button_font)
    instruction_button.place(x=710, y=175)
    new_registration_button = Button(home, text='Face Registration',command=take_image,fg=fgColor,bg=bgColor,width=15,height=1, activebackground = "snow" ,font=button_font)
    new_registration_button.place(x=710, y=275)
    attendance_button = Button(home, text='Attendance',command=attendance, fg=fgColor,bg=bgColor,width=15,height=1, activebackground = "snow" ,font=button_font)
    attendance_button.place(x=710, y=375)
    exit_button = Button(home, text='Exit', command=root.destroy,fg=fgColor,bg='red',width=15,height=1, activebackground = "snow" ,font=button_font)
    exit_button.place(x=710, y=575)

main_window()

root.mainloop()