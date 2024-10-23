<script>
    import { onMount } from "svelte";
    
    export let group = "8";
    export let source = "web";

    let images = [];

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
            });
    }

    onMount(() => {
        load_images();
    });
</script>

<div class="image_holder">
    {#each images as im}
        <div class="image">
            <img src={"/images/" + im} alt="source" />
        </div>
    {/each}
</div>

<style>
    .image {
        max-height: 400px;
        margin: 1rem;
        width: fit-content;
    }

    .image img {
        max-height: 100%;
    }

    .image_holder {
        display: flex;
        flex-direction: row;
        overflow-y: scroll;
        max-width: 80vw;
        height: 450px;
        border: 1px solid white;
    }
</style>
