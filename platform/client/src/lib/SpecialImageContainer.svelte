<script>
    import { onMount } from "svelte";

    export let group = "8";
    let source = "final";

    let images = [];

    let images_web = [];
    let images_ai = [];

    let display_popup = false;
    let compared_image = "";

    let similarities = {};

    $: group && load_images();

    function load_images() {
        fetch("/api/get-group-images", {
            method: "post",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ group: group }),
        })
            .then((res) => res.json())
            .then((data) => {
                images = data[source];
                images_ai = data["ai"];
                images_web = data["web"];
            });
    }

    function show_comparison(img) {
        display_popup = true;
        compared_image = img;
        similarities = {};
        images_web.forEach((im) => {
            fetch("/api/get-similarity", {
                method: "post",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ im1: img, im2: im }),
            })
                .then((res) => res.json())
                .then((data) => {
                    similarities[im] = data.similarity;
                });
        });
        images_ai.forEach((im) => {
            fetch("/api/get-similarity", {
                method: "post",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ im1: img, im2: im }),
            })
                .then((res) => res.json())
                .then((data) => {
                    console.log(data);
                    similarities[im] = data.similarity;
                });
        });
    }

    onMount(() => {
        load_images();
    });

    function sorted_similarities() {
        let s = Object.keys(similarities).sort((a, b) => similarities[b] - similarities[a]);
        console.log(s);
        return s;
    }
</script>

<div class="image_holder">
    {#each images as im}
        <div class="image">
            <img
                src={"/images/" + im}
                on:click={() => {
                    show_comparison(im);
                }}
                alt="source"
            />
        </div>
    {/each}
</div>

{#if display_popup}
    <div id="popup">
        <button
            on:click={() => {
                display_popup = false;
            }}>X</button
        >
        <div id="split">
            <div id="src-hold">
                <h1>Final Image</h1>
                <img src={"/images/" + compared_image} alt="source" />
            </div>
            <div id="other">
                <h3>Top 5 most similar</h3>
                <div id="imlist">
                    {#each Object.keys(similarities).sort((a, b) => similarities[b] - similarities[a]).slice(0,5) as im}
                        <div>
                            <img src={"/images/" + im} alt="source" />
                            <p>{similarities[im]}</p>
                        </div>
                    {/each}
                </div>
            </div>
        </div>
    </div>
{/if}

<style>
    .image {
        max-height: 400px;
        margin: 1rem;
        width: fit-content;
    }

    .image img {
        max-height: 100%;
        cursor: pointer;
    }

    .image_holder {
        display: flex;
        flex-direction: row;
        overflow-y: scroll;
        max-width: 80vw;
        height: 450px;
        border: 1px solid white;
    }
    #popup {
        position: absolute;
        top: 10vh;
        left: 10vw;
        width: 80vw;
        height: 80vh;
        background-color: black;
        border: 2px solid white;
        border-radius: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    #popup button{
        align-self: flex-end;
        margin-top: 60px;
        margin-right: 60px;
    }
    #split {
        display: flex;
        flex-direction: row;
        width: 100%;
        height: 100%;
    }
    #src-hold{
        min-width: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    #split img {
        max-width: 100%;
    }
    #other {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        max-height: 100%;
    }
    #imlist {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        max-height: 70%;
        overflow-y: scroll;
    }
    #imlist img {
        max-width: 90%;
    }
</style>
