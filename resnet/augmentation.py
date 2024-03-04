import numpy as np
# from PIL import Image
import threading

def rotation(address, angle):
    image = np.load(address)[0]
    displacement = len(image)//2
    rotated_image = [[0 for j in range(len(image))] for i in range(len(image))]
    rotation_matrix = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    for i in range(len(image)):
        for j in range(len(image)):
            rotation_vector = np.matmul(rotation_matrix,np.array([j-displacement,i-displacement])) #(y,x) = (i,j)
            if rotation_vector[0]>74 or rotation_vector[1]>74 or rotation_vector[0]<-74 or rotation_vector[1]<-74: continue
            rotated_image[int(rotation_vector[0])+displacement][int(rotation_vector[1])+displacement] = image[i][j]
    return np.array(rotated_image)

def translation(address, direction):
    image = np.load(address)[0]
    translated_image = np.array([[0.0 for j in range(len(image))] for i in range(len(image))])
    displacement = len(image)//2
    x0, y0 = 0, 0
    translation_scale = 30
    if direction == 1: y0 = 1
    if direction == 2: x0 = 1
    if direction == 3: y0 = -1
    if direction == 4: x0 = -1
    translation_matrix = np.array([[1,0,x0*translation_scale],[0,1,y0*translation_scale],[0,0,1]])
    for i in range(len(image)):
        for j in range(len(image)):
            translation_vector = np.matmul(translation_matrix,np.array([j-displacement,i-displacement,1])) #(y,x) = (i,j)
            if translation_vector[0]>74 or translation_vector[1]>74 or translation_vector[0]<-74 or translation_vector[1]<-74: continue
            translated_image[int(translation_vector[0])+displacement][int(translation_vector[1])+displacement] = image[i][j]
    return np.array(translated_image)

def reflection(address):
    image = np.load(address)[0]
    reflected_image = np.array([[0.0 for j in range(len(image))] for i in range(len(image))])
    

def rotation_worker(source_dir, dest_dir, angle, range_assigned, class_label):
    for i in range_assigned:
        np.save(
            dest_dir+class_label+'/rotation/%d_%d.npy'%(i+1,angle),
            rotation(
                source_dir+class_label+'/%d.npy'%(i+1),
                np.pi*(angle/4)
            )
        )

def translation_worker(source_dir, dest_dir, direction, range_assigned, class_label):
    for i in range_assigned:
        np.save(
            dest_dir+class_label+'/translation/%d_%d.npy'%(i+1,direction),
            translation(
                source_dir+class_label+'/%d.npy'%(i+1),
                direction
            )
        )

def main():
    for label in ['no','vort']:
        thread_list = []
        # for i in range(1,8):
        #     thread_list.append(threading.Thread(target=rotation_worker,args=('dataset/train/','/media/anirudh/Extreme SSD1/DeepLense/augmented_dataset/',i,range(10000),label,)))
        #     thread_list[-1].start()
        for i in range(1,5):
            thread_list.append(threading.Thread(target=translation_worker,args=('dataset/train/','augmented_dataset/',i,range(10000),label,)))
            thread_list[-1].start()
        for i in range(len(thread_list)):
            thread_list[i].join()

if __name__ == '__main__':
    main()
    # start_directory = 'dataset/train/'
    # end_directory = 'augmented_dataset/'
    # for i in ['no']:
    #     for j in range(10000):
    #         for k in [1,2,3,4,5,6,7]:
    #             np.save(
    #                 end_directory+i+'/rotation/%d_%d.npy'%(j+1,k),
    #                 rotation(
    #                     start_directory+i+'/%d.npy'%(j+1),
    #                     np.pi*(k/4)
    #                 )
    #             )
