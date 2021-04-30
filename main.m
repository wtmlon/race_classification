clc,clear;
face = imread('face.jpeg');
mean=imread('mean.jpeg');

meanIMG = zeros(218,178,3,'single');
meanIMG(:,:,:) = mean;
faceIMG = zeros(218,178,3,'single');
faceIMG(:,:,:) = face;

result = faceIMG - meanIMG;
result1 = im2uint8(result);
figure(3),
imshow(result);
figure(2),
imshow(face);
imwrite(result, '1.bmp');