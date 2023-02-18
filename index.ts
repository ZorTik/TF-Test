// TensorFlow pre-trained toxicity model test with use of Google Translate API
// Author: ZorTik
import * as toxicity from '@tensorflow-models/toxicity';
import * as readline from "readline";
import fetch from "node-fetch";
import { config } from 'dotenv';

config();

const sourceLanguage = process.env.SOURCE_LANGUAGE;
const translationApiKey = process.env.TRANSLATION_API_KEY;

const translate = (str: string) => {
    console.log("Translating...");
    return fetch(`https://translation.googleapis.com/language/translate/v2?key=${translationApiKey}`, {
        method: "POST", 
        body: JSON.stringify({
            q: str,
            source: sourceLanguage,
            target: "en",
        })
    })
        .then((res: any) => res.json())
        .then((res: any) => {
            if (res.error)
                throw "Error while translating: " + res.error.message;
        })
        .then((res: any) => res.data.translations[0].translatedText)
        .catch((err: any) => {
            console.error(err);
            return str;
        });
}

console.log("Loading toxicity model...");

toxicity.load(0.3, ["insult"]).then(async model => {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    let input = "";
    while (input !== "exit") {
        input = await new Promise(resolve => {
            rl.question("Enter a sentence to test for toxicity: ", (answer) => {
                resolve(answer);
            });
        });

        input = await translate(input);
        
        const predictions: any = (await model.classify(input)).filter(pr => {
            return pr.results[0].probabilities[1] > 0.5;
        }).map((pr: any) => {
            return {
                label: pr.label,
                result: Math.round(pr.results[0].probabilities[1] * 100)
            }
        });

        if (predictions.length > 0) {
            console.log("Toxicity detected:");
            predictions.forEach((pr: any) => {
                console.log(`: ${pr.result}% ${pr.label}`);
            });
        } else {
            console.log("No toxicity detected.");
        }
    }

    console.log("Exiting...");
    rl.close();
});