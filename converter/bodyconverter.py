import os
import SimpleITK as sitk

def convert_dicom_to_nifti(dicom_dir, output_dir):
    try:
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)

        if not series_IDs:
            print("DICOM 시리즈를 찾을 수 없어요")
            return None

        series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
        reader.SetFileNames(series_file_names)

        image = reader.Execute()
        image = sitk.Cast(image, sitk.sitkInt16) 

        output_path = os.path.join(output_dir, "converted_from_dicom.nii.gz")
        sitk.WriteImage(image, output_path)
        return output_path

    except Exception as e:
        print(f"진짜 반골 기질이 있는 친구구나 좀!: {e}")
        return None
