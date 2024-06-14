import cv2
import numpy as np
import math
import skimage

from skimage.metrics import structural_similarity as ssim

def psnr(target, ref):
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)
def mse(target, ref):
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    diff = ref_data - target_data

    err = np.sum(diff ** 2)
    err /= float(target.shape[0] * target.shape[1])

    return err

def compare_image(target, ref):
    scores = {}
    scores['psnr'] = psnr(target, ref)
    scores['mse'] = mse(target, ref)
    scores['ssim'] = ssim(target, ref, win_size=3, multichannel = True)

    return scores

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


target = cv2.imread('/home/lu13/Documents/BSDS300/images/test/hr_img/1.jpg')
sel = cv2.imread('/home/lu13/Documents/BSDS300/images/test/lr_img/1.jpg')
train = cv2.imread('/home/lu13/Documents/BSDS300/images/test/sr_img/1.jpg')
combined_img = cv2.hconcat([target, sel, train])
combined_img = cv2.resize(combined_img,(1000,320))
cv_show('target | sel | train',combined_img)
# cv2.imwrite('./val/compare.png', combined_img)
# cv2.imwrite('./val/target.png', target)
# cv2.imwrite('./val/sel.png', sel)
# cv2.imwrite('./val/train.png', train)

cv_show('target', target)
cv_show('sel', sel)
cv_show('train', train)

# print(target.shape)
# print(ref.shape)
scores_1 = compare_image(target, sel)
scores_2 = compare_image(target, train)
print('compare target and sel',scores_1)
print('compare target and train',scores_2)