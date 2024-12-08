// Import required modules
import fs from "fs";
import { AutoModel, RawImage, Tensor } from "@huggingface/transformers";

// Get the image path from command line argumentsc:\Users\SITAM MEUR\Downloads\selfie.png
const imagePath = process.argv[2];

if (!imagePath) {
  console.error("Please provide an image path as a command line argument.");
  process.exit(1);
}

// Async function to handle the process
async function processImage(imagePath) {
  try {
    // Load model and processor
    const modelId =
      "onnx-community/mediapipe_selfie_segmentation_landscape-web"; // Use other mediapipe models if needed
    const model = await AutoModel.from_pretrained(modelId, { dtype: "q4" }); // Use different dtype if needed (e.g. "fp32")

    // Load image from image path
    const image = await RawImage.read(imagePath);

    // Predict alpha matte
    const { alphas } = await model({
      pixel_values: new Tensor("uint8", image.data, [
        1,
        image.height,
        image.width,
        3,
      ]),
    });

    // Define output directory
    const outputDir = "output";

    // Check if output directory exists, if not create it
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir);
    }

    // Save output mask
    const mask = RawImage.fromTensor(alphas[0].mul(255).to("uint8"), "HWC");
    await mask.save(`${outputDir}/mask.png`);
    console.log("Mask saved successfully.");

    // Apply mask to original image
    const result = image.clone().putAlpha(mask);
    await result.save(`${outputDir}/result.png`);
    console.log("Result saved successfully.");
  } catch (error) {
    // Handle errors
    console.error("Error processing image:", error);
  }
}

// Execute the function with the provided image path
processImage(imagePath);
