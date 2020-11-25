import defopt
from pathlib import Path
import cv2
import pandas



def main(*, image_file : Path='', zoom : float=1.0):
    """Select points

    :param image_file: either a jpg or pnf image
    :param zoom: zoom factor (> 0)
    """
    if not image_file:
        raise RuntimeError("ERROR must provide image_file")

    if not image_file.exists():
        raise RuntimeError(f"ERROR file {image_file} was not found")

    # Picture path
    img = cv2.imread(str(image_file))
    img = cv2.resize(img, dsize=(int(img.shape[1]*zoom), int(img.shape[0]*zoom)))

    xpix = []
    ypix = []
    print("Click on the points and type <esc> when done.")
     
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xpix.append(x)
            ypix.append(y)
            print(x,y)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    df = pandas.DataFrame({'x': xpix, 'y': ypix})
    directory = image_file.parent
    output_csv_file = directory / Path('../characters') / Path(image_file.name.split('.')[0] + ".csv")
    print(f"writing coords to CSV file {output_csv_file}")
    df.to_csv(str(output_csv_file))


if __name__ == '__main__':
    defopt.run(main)