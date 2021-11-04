from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd

class GetBbox():
    
    def __init__(
        self,
        root_dir,
        output_file,
        pattern,
        bbox_cols=[
            'PatientID',
            'x1',
            'x2',
            'y1',
            'y2',
            'z1',
            'z2',
        ],  
    ):

        self.root_dir = Path(root_dir)
        self.pattern = pattern
    
        if pattern:
            self.patient_ids = np.unique(
                [ff.stem.split('_')[0] for ff in self.root_dir.iterdir()]
            )
        else:
            self.patient_ids = np.unique(
                [ff.stem.split('.')[0] for ff in self.root_dir.iterdir()]
            )
        
        self.output_file = Path(output_file)

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
        return data.T

    @staticmethod
    def get_bbox_from_gtv(data):
        contour = np.where(data)

        bbox = np.zeros(6)
        bbox[::2] = np.min(contour, axis=1)
        # add one pixel for numpy indexing style
        bbox[1::2] = np.max(contour, axis=1) + 1 

        return bbox


    def get_case(self, pid):
        if self.pattern:
            path = self.root_dir / (pid + '_' + self.pattern + '.nii.gz')
        else:
            path = self.root_dir / (pid + '.nii.gz')
        ds = self.get_itk_from_path(str(path))

        data = self.get_array_from_itk(ds)
        bbox_ijk = self.get_bbox_from_gtv(data)

        spacing = ds.GetSpacing()
        origin = ds.GetOrigin()

        bbox_LPS = np.zeros(6)
        bbox_LPS[::2] = self.get_coordinates(
            bbox_ijk[::2],
            origin,
            spacing
        )
        bbox_LPS[1::2] = self.get_coordinates(
            bbox_ijk[1::2],
            origin,
            spacing
        )
        return bbox_LPS

    def to_disk(self):
        if self.output_file.exists():
            raise ValueError('output file exists')
        if not self.output_file.parent.exists():
            self.output_file.parent.mkdir(parents=True)

        bbox_df = pd.DataFrame(columns=self.bbox_cols)
        idx = 0

        for pid in self.patient_ids:
            try:
                bbox = self.get_case(pid)
            except (Exception, ArithmeticError) as e:
                print(f'failed for {pid}: {e}')
                continue

            bbox_df.loc[idx] = [pid, *bbox]
            idx += 1

        bbox_df.to_csv(
            self.output_file,
            index=False
        )

if __name__ == '__main__':
    root_dir = './../data/hecktor2021_train/hecktor_nii'
    # get all files with the 'gtvt' ending
    pattern = 'gtvt'
    output_file = './../data/hecktor2021_train/hecktor2021_bbox_training_tight.csv'
    
    bbox_gen = GetBbox(
        root_dir,
        output_file,
        pattern
    )

    bbox_gen.to_disk()
