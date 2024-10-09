import torch 
from math import pi

def generate_gabor_patch(width, height, frames, ppd, fps, s_freq, t_freq, orientation, LMS_mean, LMS_delta, ge_sigma, eccentricity, phase=0, device=None):

    # I need to find a way to include the batch in all of this! To make it work for multiple batches!
    # For now, we will keep it simple!

    width_deg = width / ppd
    height_deg = height / ppd

    xx, yy = torch.meshgrid(torch.linspace(-width_deg/2, width_deg/2, width).to(device) - eccentricity , torch.linspace(-height_deg/2, height_deg/2, height).to(device), indexing='xy')
    
    rr = xx * torch.cos(torch.deg2rad(-orientation)) + yy * torch.sin(torch.deg2rad(-orientation))

    grating = torch.sin(2*pi*rr*s_freq + torch.deg2rad(torch.tensor(phase)))
    sine_grating = grating.view(height, width, 1, 1) * LMS_delta.view(1, 1, 3, 1)

    t = torch.linspace(0, frames/fps, frames).to(device)
    temporal_sine = torch.cos(2*pi*t*t_freq).view(1, 1, 1, frames) # This is giving us H,W,C,F maybe we will want to change that

    R2 = xx**2 + yy**2
    ge =  torch.exp(-R2/(2*ge_sigma**2)).view(height, width, 1, 1)  # Create the gaussian envelope

    LMS_mean = LMS_mean.view(1, 1, 3, 1)

    LMS = LMS_mean + sine_grating * ge * temporal_sine

    return LMS

def generate_gabor_mask(width, height, ppd, s_freq, orientation, LMS_mean, LMS_delta, device=None):
    
    # We are using a cos grating! 

    width_deg = width / ppd
    height_deg = height / ppd

    xx, yy = torch.meshgrid(torch.linspace(0, width_deg, width).to(device) , torch.linspace(0, height_deg, height).to(device), indexing='xy')
    
    dd = torch.sqrt(xx**2 + yy**2)
    cosB = ( xx * torch.sin(torch.deg2rad(-orientation)) + yy * torch.cos(torch.deg2rad(-orientation)) ) / dd
    cosB[0, 0] = 1
    d = torch.sqrt(1 - cosB**2) * dd 

    grating = torch.cos(2*pi*d*s_freq)
    masker_grating = grating.view(height, width, 1, 1) * LMS_delta.view(1, 1, 3, 1)

    LMS_mean = LMS_mean.view(1, 1, 3, 1)

    LMS = LMS_mean + masker_grating

    return LMS

def Yxy2XYZ(Yxy):

    # We will assume that Yxy is a tensor of the sahpe X,C

    X = Yxy[:,0] * Yxy[:,1] / Yxy[:,2]
    Y = Yxy[:, 0]
    Z = Yxy[:,0] / Yxy[:,2] * (1-Yxy[:,1]-Yxy[:,2])

    return torch.stack((X, Y, Z), dim=-1)

def XYZ2LMS2006(XYZ):

    # We expect a vector of shape X,C 

    M_xyz_lms2006 = torch.tensor([ 
        [0.187596268556126, 0.585168649077728, -0.026384263306304],
        [-0.133397430663221, 0.405505777260049, 0.034502127690364],
        [0.000244379021663, -0.000542995890619, 0.019406849066323 ]], device=XYZ.device)
    
    return torch.matmul(XYZ, M_xyz_lms2006.T)

def DKL2LMS2006_D65(DKL):
    # We expect a vector of shape X,C

    lms_gray = [0.739876529525622,   0.320136241543338,   0.020793708751515]
    mc1 = lms_gray[0]/lms_gray[1]
    mc2 = (lms_gray[0]+lms_gray[1])/lms_gray[2]

    M_lms2006_dkl = torch.tensor([
        [1, 1, 0],
        [1, -mc1, 0],
        [-1, -1, mc2]], device=DKL.device)

    M_dkl_lms2006 = M_lms2006_dkl.inverse()

    return torch.matmul(DKL, M_dkl_lms2006.T)

def LMS20062XYZ_nd(LMS):
    # We expect the input to be a multi-dimensional tensor of the shape (C,...)

    M_lms2006_xyz = torch.tensor([
        [2.629129278399650, -3.780202391780134, 10.294956387893450],
        [0.865649062438827, 1.215555811642301, -0.984175688105352],
        [0.008886561474676, 0.081612628990755, 51.371024830897888 ]], device=LMS.device)
    
    return torch.matmul(M_lms2006_xyz, LMS.reshape(3, -1)).reshape(LMS.size())

def XYZ2RGB709_nd(XYZ):
    # We expect the input to be a multi-dimensional tensor of the shape (C,...)

    M_xyz2rgb = torch.tensor([ 
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]], device=XYZ.device)

    return torch.matmul(M_xyz2rgb, XYZ.reshape(3, -1)).reshape(XYZ.size())

def XYZ2RGB2020_nd(XYZ):
    # We expect the input to be a multi-dimensional tensor of the shape (C,...)

    M_xyz2rgb = torch.tensor([
        [0.636953507, 0.144619185, 0.168855854],
        [0.262698339, 0.678008766, 0.0592928953],
        [4.99407097e-17, 0.0280731358, 1.06082723]], device=XYZ.device).inverse()

    return torch.matmul(M_xyz2rgb, XYZ.reshape(3, -1)).reshape(XYZ.size())

def lin2srgb(L):

    t = 0.0031308
    a = 0.055
    L = L.clip(min=0, max=1)
    p = torch.where(L <= t, L * 12.92, (1 + a) * L ** (1 / 2.4) - a)
    return p

def tensor_to_numpy_image(T):
    return torch.squeeze(T.permute((3,4,1,0,2)), dim=(3,4)).cpu().numpy()