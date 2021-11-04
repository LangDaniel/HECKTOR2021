from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd
import h5py


class GetPatches():
    
    def __init__(
        self,
        root_dir,
        bbox_file,
        output_file,
        rescale_dict,
        spacing,
        min_size,
        modalities = [
            'ct',
            'pt',
        ],
        bbox_cols=[
            'x1',
            'x2',
            'y1',
            'y2',
            'z1',
            'z2',
        ],  
    ):

        self.root_dir = Path(root_dir)
        self.patient_ids = np.unique(
            [ff.stem.split('_')[0] for ff in self.root_dir.iterdir()]
        )
        self.output_file = Path(output_file)

        self.rescale_dict = rescale_dict
        self.spacing = spacing
        self.min_size = min_size

        self.modalities = modalities

        self.bbox_df = pd.read_csv(bbox_file)
        self.bbox_cols = bbox_cols

    ######################## IO stuff ##########################

    @staticmethod
    def get_coordinates(pix_idx, origin, spacing):
        pix_idx = np.array(pix_idx)
        origin = np.array(origin)
        spacing = np.array(spacing)
        return (pix_idx * spacing) + origin

    @staticmethod
    def get_itk_from_path(path):
        data = sitk.ReadImage(path)
        
        return data

    @staticmethod
    def get_array_from_itk(image):
        data = sitk.GetArrayFromImage(image)
        return np.moveaxis(data, 0, -1)

    @staticmethod
    def get_pix_idx(coordinates, origin, spacing):
        coordinates = np.array(coordinates)
        origin = np.array(origin)
        spacing = np.array(spacing)
        return ((coordinates - origin) / spacing).astype(int)
        
    #################### read bbox #############################

    def get_bbox_LPS(self, pid):
        row = self.bbox_df.set_index('PatientID').loc[pid]
        bbox =  row[self.bbox_cols].values

        if self.min_size:
            bbox = self.fit_bbox(bbox)

        return bbox

    def fit_bbox(self, bbox):
        for jj in range(0, 3): 
            diff = bbox[1::2][jj] - bbox[::2][jj]
            margin = self.min_size[jj] - diff
            margin = np.clip(margin, a_min=0, a_max=None) 

            bbox[::2][jj] -= margin / 2 
            bbox[1::2][jj] += margin / 2 

        return bbox

    ######################### crop ##############################

    def resample_img(self, itk_image, is_label=False):
        '''
        resample image have a voxel size of self.spacing
        '''
        
        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()
    
        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / self.spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / self.spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / self.spacing[2])))]
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
    
        return resample.Execute(itk_image)

    def zero_pad(self, itk_img, bbox, const=0):
        size = np.array(itk_img.GetSize())
        spacing = np.array(itk_img.GetSpacing())
    
        img_lower = itk_img.GetOrigin()
        img_upper =  img_lower + size * spacing
    
        bb_lower = bbox[::2]
        bb_upper = bbox[1::2]
    
        lower_pad = np.zeros(3)
        upper_pad = np.zeros(3)
        for ii in range(0, 3): 
            lower_diff = img_lower[ii] - bb_lower[ii]
            if lower_diff > 0:
                lower_pad[ii] = np.ceil(lower_diff / spacing[ii]).astype(np.int)
    
            upper_diff = bb_upper[ii] - img_upper[ii]
            if upper_diff > 0:
                upper_pad[ii] = (np.ceil(upper_diff / spacing[ii])).astype(np.int)

        if not lower_pad.any() and not upper_pad.any():
            return itk_img
        print('zero padding')
        # convert to list due to sitk bug
        lower_pad = lower_pad.astype('int').tolist()
        upper_pad = upper_pad.astype('int').tolist()

        return sitk.ConstantPad(itk_img, lower_pad, upper_pad, constant=const)

    @staticmethod
    def rescale_linear(data, scaling):
        [in_low, in_high], [out_low, out_high] = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        data = m * data + b
        data = np.clip(data, out_low, out_high)
        return data

    @staticmethod
    def crop_patch(data, bbox):
        xii, xff, yii, yff, zii, zff = bbox.astype(int)
        return data[xii:xff, yii:yff, zii:zff]
    
    def get_idx_bbox(self, bbox, origin, spacing):
        bbox_idx = np.zeros(6)
        bbox_idx[::2] = self.get_pix_idx(bbox[::2], origin, spacing)
        bbox_idx[::2] = np.clip(bbox_idx[::2], a_min=0, a_max=None)
        bbox_idx[1::2] = self.get_pix_idx(bbox[1::2], origin, spacing)
        return bbox_idx.astype(int)

    def get_case(self, pid):
        bbox_LPS = self.get_bbox_LPS(pid)
        data = {}
        
        for mod in self.modalities:
            path = self.root_dir / (pid + '_' + mod + '.nii.gz')
            try:

                ds = self.get_itk_from_path(str(path))
                if mod == 'gtvt':
                    is_label = True
                else:
                    is_label = False
                ds = self.resample_img(ds, is_label=is_label)
                    
                ds = self.zero_pad(ds, bbox_LPS)
            
                bbox_idx = self.get_idx_bbox(
                    bbox_LPS,
                    ds.GetOrigin(),
                    ds.GetSpacing()
                )
                
                ds = self.crop_patch(ds, bbox_idx)
                img = self.get_array_from_itk(ds)
                if mod in self.rescale_dict.keys():
                    scaling = self.rescale_dict[mod]
                    img = self.rescale_linear(img, scaling)
            except:
                raise ValueError(f'{pid} modality {mod} failed')
            data[mod] = img

        return data

    def to_disk(self):
        if self.output_file.exists():
            raise ValueError('output file exists')
        if not self.output_file.parent.exists():
            self.output_file.parent.mkdir(parents=True)

        disk_df = pd.DataFrame(columns=['identifier'])
        disk_idx = 0

        with h5py.File(self.output_file, 'w') as ff:
            for pid in self.patient_ids:
                try:
                    data = self.get_case(pid)
                except:
                    continue

                patient_grp = ff.create_group(pid)

                for mod, array in data.items():
                    patient_grp.create_dataset(
                        mod,
                        data=array
                    )

                disk_df.loc[disk_idx] = [pid]
                disk_idx += 1

        disk_df.to_csv(
            self.output_file.parent / (self.output_file.stem + '.csv'),
            index=False
        )

if __name__ == '__main__':
    root_dir = './../data/hecktor2021_train/hecktor_nii'
    bbox_file = './../data/hecktor2021_train/hecktor2021_bbox_training_tight.csv'
    output_file = './../data/patch_data/HECKTOR_train_patches.h5'
    modalities = ['ct', 'pt']

    rescale_dict = {
        'ct': [[-250., 250.], [0., 255.]],
        'pt': [[0., 100.], [0., 255.]],
    }
    spacing = (1.0, 1.0, 1.0,)
    min_size = (100, 100, 100)
    
    patch_gen = GetPatches(
        root_dir,
        bbox_file,
        output_file,
        rescale_dict,
        spacing,
        min_size,
        modalities=modalities
    )

    patch_gen.to_disk()

    output_file = Path(output_file)
    script = output_file.parent / (output_file.stem + '_script.py')
    with open(script, 'w') as ff: 
        ff.write(Path(__file__).read_text())
