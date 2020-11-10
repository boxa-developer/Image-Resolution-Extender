import cv2
import timeit
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('./models/LapSRN_x4.pb')
sr.setModel('lapsrn', 4)

image = cv2.imread('input/image3.png')
cv2.imwrite('./output4/original.jpg', image)
start = timeit.default_timer()
res8 = sr.upsample(image)
end = timeit.default_timer()
cv2.imwrite('./output4/res8.jpg', res8)
print('Time: ', end-start)

resOrd = cv2.resize(image, (res8.shape[1], res8.shape[0]),
                    interpolation=cv2.INTER_CUBIC)

cv2.imwrite('./output4/resord.jpg', resOrd)
print('Finished!')
