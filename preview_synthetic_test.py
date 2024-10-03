"""
This mainly for testing purposes
I will be calling some of the tests, and previewing them!
"""
from pyst.GaborPatch import *
from pyst.tests_utils import get_all_tests

Tests = get_all_tests()
output_folder = 'previews'

#Tests = [GaborPatchLuminanceAchTest, GaborPatchSpatialFreqAchTest]

for T in Tests:
    Test = T()
    for i in range(len(Test)):

        full_output_folder = os.path.join(output_folder, Test.get_preview_folder())

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        test_filename = os.path.join(full_output_folder, 'test_'+str(i)) # We probably want to make this as a private variable and make a class to call it!
        ref_filename = os.path.join(full_output_folder, 'ref_' + str(i))

        if Test.get_condition_file_format() == 'image':
            test_filename += '.png'
            ref_filename += '.png'
            Test.save_as_png(i, test_filename, ref_filename)

        elif Test.get_condition_file_format() == 'video':
            test_filename += '.mp4'
            ref_filename += '.mp4'
            Test.save_as_video(i, test_filename, ref_filename)

