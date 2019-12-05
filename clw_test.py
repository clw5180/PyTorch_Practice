import os

folder_path = 'C:/Users/62349/Desktop/train/'

file_names = os.listdir(folder_path)
for idx, file_name in enumerate(file_names):
    b1 = file_name.find('fliph')
    b2 = file_name.find('flipv')
    b3 = file_name.find('rotate')
    if b1 != -1 or b2 != -1 or b3 != -1:
        os.remove(os.path.join(folder_path, file_name))
        print('delete %d file: %s' % (idx+1, file_name))