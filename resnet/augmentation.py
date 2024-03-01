import numpy as np
# from PIL import Image
import threading

def rotation(address, angle):
    image = np.load(address)[0]
    displacement = len(image)//2
    rotated_image = [[0 for i in range(len(image))] for i in range(len(image))]
    rotation_matrix = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    for i in range(len(image)):
        for j in range(len(image)):
            rotation_vector = np.matmul(rotation_matrix,np.array([j-displacement,i-displacement])) #(y,x) = (i,j)
            if rotation_vector[0]>74 or rotation_vector[1]>74: continue
            print(int(rotation_vector[0])+displacement,int(rotation_vector[1])+displacement)
            print(i,j)
            rotated_image[int(rotation_vector[0])+displacement][int(rotation_vector[1])+displacement] = image[i][j]
    return np.array(rotated_image)

if __name__ == '__main__':
    start_directory = '../dataset/train/'
    end_directory = '../augmented_data/'
    for i in ['no','sphere','vort']:
        for j in range(1):
            for k in [1,2,3,4,5,6,7]:
                np.save(
                    end_directory+i+'/rotation/%d_%d.npy'%(j+1,k),
                    rotation(
                        start_directory+i+'/%d.npy'%(j+1),
                        np.pi*(k/4)
                    )
                )
