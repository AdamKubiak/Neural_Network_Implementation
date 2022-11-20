import numpy as np

def DataGather():
    x = np.load('dataset\\full_numpy_bitmap_broccoli.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    X_train = x
    y_train = y
    X_valid = x1
    y_valid = y1

    x = np.load('dataset\\full_numpy_bitmap_cruise_ship.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,1)


    y = np.add(y,1)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)


    x = np.load('dataset\\full_numpy_bitmap_angel.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,2)


    y = np.add(y,2)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_bicycle.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,3)


    y = np.add(y,3)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_umbrella.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,4)



    y = np.add(y,4)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_octopus.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,5)



    y = np.add(y,5)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_house_plant.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,6)



    y = np.add(y,6)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_windmill.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,7)



    y = np.add(y,7)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_airplane.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,8)



    y = np.add(y,8)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_popsicle.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,9)



    y = np.add(y,9)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_axe.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,10)



    y = np.add(y,10)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_rainbow.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,11)



    y = np.add(y,11)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_envelope.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,12)



    y = np.add(y,12)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_eye.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,13)



    y = np.add(y,13)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_donut.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,14)



    y = np.add(y,14)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_lightning.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,15)



    y = np.add(y,15)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_smiley_face.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,16)



    y = np.add(y,16)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_helicopter.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,17)



    y = np.add(y,17)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_sun.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,18)



    y = np.add(y,18)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    x = np.load('dataset\\full_numpy_bitmap_dolphin.npy')
    x1 = x[20000:21000]
    y1 = np.zeros(1000)
    x = x[:20000]
    y = np.zeros(20000)
    y1 = np.add(y1,19)



    y = np.add(y,19)
    X_train = np.append(X_train,x,axis = 0)
    y_train = np.append(y_train,y)
    X_valid = np.append(X_valid,x1,axis =0)
    y_valid = np.append(y_valid,y1)

    y_train = y_train.astype(int)
    y_valid = y_valid.astype(int)
    #X_train = ((X_train / 255.)-.5)*2
    #X_valid = ((X_valid / 255.)-.5)*2
    X_train = X_train / 255.
    X_valid = X_valid / 255.
    np.save('Dane\\X_train.npy', X_train)
    np.save('Dane\\y_train.npy', y_train)
    np.save('Dane\\X_valid.npy', X_valid)
    np.save('Dane\\y_valid.npy', y_valid)

