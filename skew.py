import numpy as np
import cv2

one_deg_in_rad = 1 / 180 * np.pi

def plot_lines(img, lines):
    img = img.copy()
    for r, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 - 1000 * b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 + 1000 * b)
        y2 = int(y0 - 1000 * a)
        
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    return img

def get_lines(bimg, threshold=700):
    lines = cv2.HoughLines(bimg, 1, np.pi / 180, threshold)
    lines = lines.reshape((len(lines), 2))
    thetas = lines[:, 1]
    phis = np.pi / 2 - thetas
    idx = (phis > - np.pi / 4) & (phis < np.pi / 4)
    
    return lines[idx]
    
def get_angle(bimg, bias=0):
    
    lines = get_lines(bimg)
    thetas = lines[:, 1]
    phis = np.pi / 2 - thetas
    
    '''
    left_phis = phis[phis > 0]
    right_phis = phis[phis < 0]
    '''
    bins = np.linspace(- np.pi /4, np.pi / 4, 451)
    freq, _ = np.histogram(phis, bins)
    print(freq, bins)
    idx = np.argmax(freq)
    angle = bins[idx] + bins[1] - bins[0]
    
    '''
    if len(left_phis) > len(right_phis):
        angle = left_phis.mean()
    else:
        angle = right_phis.mean()
    '''
    
    return angle + bias

def binarize(img, threshold = 150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    idx = gray < threshold
    bimg = gray.copy()
    bimg[idx] = 0
    bimg[~idx] = 255
    return 255 - bimg

def rotate(img, angle):
    angle = int(angle / np.pi * 180)
    height, width, *_ = img.shape
    m = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
    res = cv2.warpAffine(img, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return res

def skew(img):
    bimg = binarize(img)
    #lines = get_lines(bimg)
    #limg = plot_lines(img, lines)
    #cv2.imwrite('lines.jpg', limg)
    
    #return
    angle = get_angle(bimg)
    print(angle / np.pi * 180)
    res = rotate(img, -angle)
    return res

def main(fname):
    img = cv2.imread(fname)
    res = skew(img)
    cv2.imwrite('res.jpg', res)
    print('done..')
    

if __name__ == '__main__':
    main('20180530155554.jpg')