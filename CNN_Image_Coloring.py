import os
import cv2
import numpy as np
import webbrowser


def CNN():
    # Lendo os arquivos das imagens
    frames = []
    for i in range(1,14):
        print("Teste {}:".format(i))
        path = "input/{}.jpg".format(i)
        frame = cv2.imread(path)

        # Especificando os modelos e os arquivos com varias versões dos pesos
        protoFile = "./datasets/colorization_deploy_v2.prototxt"
        weightsFiles = {
            'v1':"./datasets/colorization_release_v1.caffemodel",
            'v2':"./datasets/colorization_release_v2.caffemodel",
            'v2.norebal':"./datasets/colorization_release_v2_norebal.caffemodel"
        }

        for version, model in weightsFiles.items():
            weightsFile = model

            # Carregando o centro do cluster
            pts_in_hull = np.load('./bin/pts_in_hull.npy')

            # Carregando a Rede Neural
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

            # Populando os centros dos clusters com uma núcleo convolucional 1x1
            pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
            net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
            net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

            # from opencv sample
            W_in = 224
            H_in = 224

            # Conversão das imagens de RGB pata LAB e separando canal L
            img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
            img_l = img_lab[:,:,0] # pull out L channel

            # Redimencionando o canal L (lightness) dos tamanhho dos inputs da rede
            img_l_rs = cv2.resize(img_l, (W_in, H_in)) #
            img_l_rs -= 50 # Subtraindo 50  para o media dos centros

            # Carregando a CNN do tipo Blob e Coletando os Resultados
            net.setInput(cv2.dnn.blobFromImage(img_l_rs))
            ab_dec = net.forward()[0,:,:,:].transpose((1,2,0))

            # Redimencionando as imagens para os tamanhos originais
            (H_orig,W_orig) = img_rgb.shape[:2]
            ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
            # Convertendo os resultados de Lab pata RGB
            img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
            img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

            outputFile = 'output/{}_{}_colorized.jpg'.format(i,version)
            cv2.imwrite(outputFile, (img_bgr_out*255).astype(np.uint8))
            print('\tColorizada imagem {} e salva como {}'.format(i,outputFile))
    print('CNN excutada com sucesso!')

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        print("Você precisa baixar os datasets no link")
        webbrowser.open('https://drive.google.com/open?id=1NiILe6-0S6ZQN2sK7c74Hc5Gj4j2W4Au')
    elif len(os.listdir("datasets") ) == 0:
        print("Baixar os datasets no link")
        webbrowser.open('https://drive.google.com/open?id=1NiILe6-0S6ZQN2sK7c74Hc5Gj4j2W4Au')
    else:
        CNN()
