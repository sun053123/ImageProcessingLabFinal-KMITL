from ImageManager import ImageManager
from ImageManager2 import ImageManager2
from ImageManager2 import StructuringElement


# นาย ภูผา ศิริโกมลสิงห์ 61050273 \ Image Processing LAB \

def main():
    # เริ่มโปรแกรม
    print("program 61050273 imgage processing LAB start here... \n")
    img2 = ImageManager2()

    img2.read("images/images/4/motion01.512.bmp")
    if img2 is None:
        print("import failure")
        return
    print("image import successful ")

    seq = [
        "images/images/4/motion02.512.bmp",
        "images/images/4/motion03.512.bmp",
        "images/images/4/motion04.512.bmp",
        "images/images/4/motion05.512.bmp",
        "images/images/4/motion06.512.bmp",
        "images/images/4/motion07.512.bmp",
        "images/images/4/motion08.512.bmp",
        "images/images/4/motion09.512.bmp",
        "images/images/4/motion10.512.bmp",
    ]

    img2.ADIAbsolute(seq,25,50)

    img2.write("images/ADIA.bmp")


if __name__ == "__main__":
    main()
