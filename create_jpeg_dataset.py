import os 
import shutil

in_dataset = '/local/scratch/rkm38/upiq_dataset/images/tid2013/'
out_dataset = 'jpeg_dataset/'

indices = ['03', '15', '21', '23']

for index in indices:
    in_image = os.path.join(in_dataset, index, 'i'+index+'.png')
    out_ref_image = os.path.join(out_dataset, 'ref_images', 'i'+index+'.png')
    shutil.copyfile(in_image, out_ref_image)

    # Convert from PNG to PPM using JPEG
    cmd = f'convert {in_image} {out_dataset}/ppm_ref_images/i{index}.ppm'
    os.system(cmd)

    # JPEG encoding
    for q in range(101):
        out_test_image = os.path.join(out_dataset, 'test_images', 'i'+index+'_'+str(q)+'.jpeg')
        cmd = f'cjpeg -quality {q} {out_dataset}/ppm_ref_images/i{index}.ppm > {out_test_image}'
        os.system(cmd)