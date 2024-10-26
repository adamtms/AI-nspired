import { useEffect, useState } from "react";
import { get_images } from "../lib/data";
import ComparisonPopup from "./ComparisonPopup";

interface FinalImageContainerProps {
    source: string;
    group: string;
}

export default function FinalImageContainer({
    source, group
}: FinalImageContainerProps) {

    const [images, setImages] = useState<string[]>([]);
    const [selectedImage, setSelectedImage] = useState<string>("");
    const [popupOpen, setPopupOpen] = useState<boolean>(false);

    useEffect(() => {
        get_images(group, source).then((images) => {
            setImages(images);
        });
    }, [group, source]);

    function handleImageClick(image: string) {
        setSelectedImage(image);
        setPopupOpen(true);
    }


    return (
        <>
            <div className="image-container">
                {images.map((image) => (
                    <div className="image">
                        <img className="clickable" src={`${process.env.REACT_APP_BACKEND_URL}/images/` + image} alt="source" onClick={() => {handleImageClick(image)}}/>
                    </div>
             
             ))}
            </div>
            <ComparisonPopup popup_open={popupOpen} set_popup_open={setPopupOpen} selectedGroup={group} selectedImage={selectedImage}/>
        </>
    )
}