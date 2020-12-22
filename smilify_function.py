import cv2
import boto3
import os
import numpy as np
from io import BytesIO
from PIL import Image
import boto3.exceptions as Botexc

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')


def detection(grayscale, img):
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face + h_face, x_face:x_face + w_face]
        ri_color = img[y_face:y_face + h_face, x_face:x_face + w_face]
        eye = cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18)
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            cv2.rectangle(ri_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)
        smile = cascade_smile.detectMultiScale(ri_grayscale, 1.7, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile:
            cv2.rectangle(ri_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)
    return img


# elasticbeanstalk-eu-west-1-571959335296
# IMG_5589.JPG
class Imager:
    bucket = ''
    key = ''
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = os.getenv('AWS_DEFAULT_REGION')
    s3 = None

    def __init__(self, bucket, key):
        self.bucket = bucket
        self.key = key
        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )
        s3 = session.resource('s3')
        self.s3 = s3

    def resource(self):
        pass

    def getImage(self):
        obj = self.s3.Bucket(self.bucket)
        file_content = obj.Object(self.key).get()['Body'].read()
        np_array = np.frombuffer(file_content, np.uint8)
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image_np

    def checkFaceUpload(self):
        img = self.getImage()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = detection(grayscale, img)
        cv2.imwrite('me.jpg', result);
        image = Image.fromarray(result, mode='RGB')
        in_mem_file = BytesIO()
        key = self.key + 'smiles.jpg'
        image.save(in_mem_file, format='JPEG')
        image.save('test.png', format='PNG')
        in_mem_file.seek(0)

        try:
            # self.s3.Bucket(self.bucket).Object(self.key+'smiles.jpg').upload_fileobj(BytesIO(result), {'ContentType':'image/jpg'})
            self.s3.Bucket(self.bucket).upload_fileobj(in_mem_file, key, {'ContentType': 'image/jpg'})

        except(Botexc.Boto3Error, Botexc.S3UploadFailedError, ValueError) as e:
            print(e)
            print(e.args)

        BytesIO().close()
        return "https://{bucket}.s3.amazonaws.com/{resized_key}".format(bucket=self.bucket, resized_key=key)

    def showResult(self):
        img = self.getImage()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Smile", detection(grayscale, img))


# make = Imager('elasticbeanstalk-eu-west-1-571959335296', 'IMG_5589.JPG')
# make.checkFaceUpload()


def smilify_function(event, context):
    # bucket = event['queryStringParameters'].get('bucket', None)
    # key = event['queryStringParameters'].get('key', None)
    bucket = event['bucket']
    key = event['key']

    imager = Imager(bucket, key)
    image_s3_url = imager.checkFaceUpload()

    return {
        'statusCode': 301,
        'body': image_s3_url
    }
# make.showResult()

# img = imgRead('elasticbeanstalk-eu-west-1-571959335296', 'IMG_5589.JPG')

# img = cv2.imread('IMG_5589.JPG');
# # cv2.imshow("Test", img)
# grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.imshow("Test", grayscale)
# final = detection(grayscale, img)
# cv2.imshow('Smile', detection(grayscale, img))
# cv2.waitKey(0)

# closing all open windows
# cv2.destroyAllWindows()
