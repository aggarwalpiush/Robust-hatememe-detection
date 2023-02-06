import augly.image as imaugs
from PIL import Image
import augly.text as txtaugs

# path = '01284.png'
# image = Image.open(path)

# image_transform = imaugs.OverlayStripes()
# image = image_transform(image)
# image.save('stripes.png')

# [imaugs.Blur(), imaugs.RandomNoise(), imaugs.ShufflePixels(), imaugs.ColorJitter(), imaugs.OverlayStripes()]


text = 'muslims offend me'
list_augs = [txtaugs.ChangeCase(granularity='char'), 
             txtaugs.ReplaceBidirectional(), 
             txtaugs.ReplaceSimilarChars(), 
             txtaugs.SimulateTypos()]

for aug in list_augs:
    text_aug = aug(text)
    print(text_aug)
                