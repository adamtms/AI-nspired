import { useEffect, useState } from "react";
import { get_images } from "../lib/data";
import "./ImageContainer.css";

interface ImageContainerProps {
    selectedGroup: string;
    source: string;
}

export default function ImageContainer({ selectedGroup, source }: ImageContainerProps) {
    const [images, setImages] = useState<string[]>([]);

    useEffect(() => {
        get_images(selectedGroup, source).then((images) => {
            setImages(images);
        });
    }, [selectedGroup, source]);

    return (
        <>
            <div className="image-container">
                {images.map((image) => (
                    <div className="image">
                        <img src={`${process.env.REACT_APP_BACKEND_URL}/images/` + image} alt="source" />
                    </div>
                ))}
            </div>
        </>
    )
}