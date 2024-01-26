import cv2
import face_recognition

#Masukan Foto
fotoSatu = cv2.imread("elonmusk.jpg")
RGB_fotoSatu = cv2.cvtColor(fotoSatu,cv2.COLOR_BGR2RGB)
enkode_wajahSatu = face_recognition.face_encodings(RGB_fotoSatu)[0]

fotoDua = cv2.imread("jeffbezos.webp")
RGB_fotoDua = cv2.cvtColor(fotoDua,cv2.COLOR_BGR2RGB)
enkode_wajahDua = face_recognition.face_encodings(RGB_fotoDua)[0]

#Mendeteksi Lokasi Muka
lokasi_wajahSatu = face_recognition.face_locations(fotoSatu)[0]
cv2.rectangle(fotoSatu,(lokasi_wajahSatu[3],lokasi_wajahSatu[0]),(lokasi_wajahSatu[1],lokasi_wajahSatu[2]),(255,0,255),2)

lokasi_wajahDua = face_recognition.face_locations(fotoDua)[0]
cv2.rectangle(fotoDua,(lokasi_wajahDua[3],lokasi_wajahDua[0]),(lokasi_wajahDua[1],lokasi_wajahDua[2]),(255,0,255),2)

#Menampilkan Hasil
hasilPerbandingan = face_recognition.compare_faces([enkode_wajahSatu], enkode_wajahDua)
numerikWajah = face_recognition.face_distance([enkode_wajahSatu], enkode_wajahDua)
print(hasilPerbandingan, numerikWajah)
cv2.putText(fotoSatu,f"{hasilPerbandingan} {round(numerikWajah[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.putText(fotoDua,f"{hasilPerbandingan} {round(numerikWajah[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

#Menampilkan Foto
cv2.imshow("Foto Satu", fotoSatu)
cv2.imshow("Foto Dua", fotoDua)
cv2.waitKey(0)