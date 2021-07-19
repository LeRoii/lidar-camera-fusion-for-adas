import math
import numpy as np

ellipse_a  = 6378137
ellipse_e  = 0.081819190842622

basePoint = [34.2569999, 108.6511768, 392.931]
mat = np.zeros((4,3), dtype='float')

def generateMat(lat,lon,height):
    lat = lat / 180 * math.pi
    lon = lon / 180 * math.pi
    sinLati = math.sin(lat)
    cosLati = math.cos(lat)
    sinLong = math.sin(lon) 
    cosLong = math.cos(lon)

    N = ellipse_a / math.sqrt((1-ellipse_e*ellipse_e*sinLati*sinLati))

    mat[1][0] = -sinLong
    mat[1][1] = cosLong
    mat[1][2] = 0
    mat[0][0] = -sinLati*cosLong
    mat[0][1] = -sinLati*sinLong
    mat[0][2] = cosLati
    mat[2][0] = cosLati*cosLong
    mat[2][1] = cosLati*sinLong
    mat[2][2] = sinLati

    mat[3][0] = (N + height)*cosLati*cosLong
    mat[3][1] = (N + height)*cosLati*sinLong
    mat[3][2] = (N*(1 - ellipse_e*ellipse_e) + height)*sinLati

    return mat

def gps2ENU(lat,lon,height):
    x0 = mat[3][0]
    y0 = mat[3][1]
    z0 = mat[3][2]

    lat = lat / 180 * math.pi
    lon = lon / 180 * math.pi
    sinLati = math.sin(lat)
    cosLati = math.cos(lat)
    sinLong = math.sin(lon) 
    cosLong = math.cos(lon)

    N = ellipse_a / math.sqrt((1-ellipse_e*ellipse_e*sinLati*sinLati))
    x1 = (N + height)*cosLati*cosLong
    y1 = (N + height)*cosLati*sinLong
    z1 = (N*(1 - ellipse_e*ellipse_e) + height)*sinLati
    dx = x1 - x0 
    dy = y1 - y0 
    dz = z1 - z0
    outputy = mat[0][0] * dx + mat[0][1] * dy + mat[0][2] * dz
    outputx = mat[1][0] * dx + mat[1][1] * dy
    outputz = mat[2][0] * dx + mat[2][1] * dy + mat[2][2] * dz

    return outputx,outputy,outputz


if __name__ == "__main__":
    mat = generateMat(basePoint[0], basePoint[1], basePoint[2])
    print('mat:',mat)