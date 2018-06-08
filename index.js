const natural = require("natural");
const classifier = new natural.BayesClassifier();
var TfIdf = natural.TfIdf;
var tfidf = new TfIdf();

const tfidfVector = (featureVector, termArray) => {
    const x = featureVector.map(vector => {
        if(termArray.indexOf(vector) != -1 ){
            return 1;
        } else 
            return 0;
    });
    return x;
}

const trainingData = [
	{
	  "author": 'john-green',
 	  "quotes": [
            'As he read, I fell in love the way you fall asleep: slowly, and then all at once.',
            'My thoughts are stars I cannot fathom into constellations.',
            'Some infinities are bigger than other infinities.',
            'The marks humans leave are too often scars.',
            'Grief does not change you, Hazel. It reveals you.',
            'The world is not a wish-granting factory.',
            'Books so special and rare and yours that advertising your affection feels like a betrayal.',
            'At some point, you just pull off the Band-Aid, and it hurts, but then it is over and you are relieved.',
            'Sometimes you lose a battle. But mischief always wins the war',
            'We need never be hopeless because we can never be irreperably broken.',  
		]
    },
    {
        'author': 'haruki-murakami',
        'quotes' : [
            'What a terrible thing it is to wound someone you really care for and to do it so unconsciously.',
            'What makes us the most normal," said Reiko, "is knowing that we are not normal.',
            'I wonder what ants do on rainy days?',
            'When your feelings build up and harden and die inside, then you are in big trouble.',
            'Just remember, life is a box of cookies.',
            'Her cry was the saddest sound of orgasm that I had ever heard.',
            'Like the wind passing over my body, it had neither shape nor weight, nor could I wrap myself in it.',
            'it feels as if we are the only ones in the world. I wish it would just keep raining so the three of us could stay together.',
            'What makes us the most normal," said Reiko, "is knowing that we are not normal.',
            'Memories warm you up from the inside. But they also tear you apart.',
            'If you remember me, then I do not care if everyone else forgets.',
            'It is like Tolstoy said. Happiness is an allegory, unhappiness a story.',
            'Taking crazy things seriously is a serious waste of time.',
            'In everybody’s life there’s a point of no return. And in a very few cases, a point where you can’t go forward anymore. And when we reach that point, all we can do is quietly accept the fact. That’s how we survive.',
            'Chance encounters are what keep us going.',
            'When you come out of the storm, you won’t be the same person who walked in. That’s what this storm’s all about.'
        ]
    }
];

const testData = [
    {'quote':'The only way out of the labyrinth of suffering is to forgive.', 'actual_author': 'john-green'},
    {'quote': 'What a slut time is. She screws everybody.', 'actual_author': 'john-green'},
    {'quote': 'Listen up - there is no war that will end all wars.', 'actual_author': 'haruki-murakami'},
    {'quote' : 'Silence, I discover, is something you can actually hear.', 'actual_author': 'haruki-murakami'},
    {'quote' : 'A certain type of perfection can only be realized through a limitless accumulation of the imperfect.', 'actual_author': 'haruki-murakami'},
];

trainingData.forEach(data => {
    data.quotes.forEach(quote => {
        classifier.addDocument(quote, data.author);
        tfidf.addDocument(quote, data.author);
    });
});
// console.log(tfidf);
classifier.train();
// testData.forEach(data => console.log( `Actual: ${data.actual_author} Result:: ${classifier.classify(data.quote)}`) );

let featureVector = []
let tfidfMatrix = [];

tfidf.documents.forEach(doc => {
    const keys = Object.keys(doc);
    keys.forEach(key => {
        if(featureVector.indexOf(key) === -1){
            featureVector.push(key);
        }
    });
});

tfidf.documents.forEach(doc => {
    const keys = Object.keys(doc);
    tfidfMatrix.push(tfidfVector(featureVector, keys));
});


console.log(tfidfMatrix);


