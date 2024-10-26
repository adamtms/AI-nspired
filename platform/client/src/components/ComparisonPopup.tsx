import { useEffect, useState } from "react";
import { load_images_to_compare } from "../lib/data";
import "./ComparisonPopup.css";

interface ComparisonPopupProps {
    selectedGroup: string;
    selectedImage: string;
    popup_open: boolean;
    set_popup_open: React.Dispatch<React.SetStateAction<boolean>>;
  }

export default function ComparisonPopup({
    selectedGroup,
    selectedImage,
    popup_open,
    set_popup_open,
}:  ComparisonPopupProps,) {
    const [similarities, setSimilarities] = useState<{ image: string; similarity: number }[]>([]);
    const [images_to_compare, setImagesToCompare] = useState<string[]>([]);

    async function load_similarities() {
        setSimilarities(() => []);
        for (const image of images_to_compare) {
          await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/get-similarity`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ im1: selectedImage, im2: image }),
          })
            .then((response) => response.json())
            .then((data) => {
              setSimilarities((oldState) => [...oldState, { image: image, similarity: data.similarity }]);
            });
        }
      }

    const closePopup = () => {
      set_popup_open(false);
    }

    useEffect(()=> {
        load_images_to_compare(selectedGroup).then((images) => {
            setImagesToCompare(() => images);
        });
    }, [selectedGroup])

    useEffect(() => {
        if(images_to_compare && selectedImage !== "") {
            load_similarities();
        }
    }, [images_to_compare, selectedImage]);

    if (!popup_open) {
        return null;
      }
      return (
        <>
          <div id="popup">
            <button
              onClick={() => {
                closePopup();
              }}
            >
              Close
            </button>
            <div className="split-horizontal">
              <div className="image-holder">
                <img src={`${process.env.REACT_APP_BACKEND_URL}/images/` + selectedImage} alt="source" />
              </div>
              <div id="similar-images">
                <h2>Top 5 most similar images</h2>
                <div className="scroll-vertical">
                  <div className="scroll-vertical">
                    {similarities
                      .sort((a, b) => b.similarity - a.similarity)
                      .slice(0, 5)
                      .map((similarity, index) => {
                        return (
                          <div key={index} className="similarity-image">
                            <img
                              src={`${process.env.REACT_APP_BACKEND_URL}/images/` + similarity.image}
                              alt="similar"
                            />
                            <p>Similarity: {similarity.similarity}</p>
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      );

}