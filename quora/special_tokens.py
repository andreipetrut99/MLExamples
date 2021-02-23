import spacy

DESIRED_SPECIAL_TOKENS = ['ORG', 'MONEY', 'DATE']

nlp = spacy.load('en_core_web_sm')
text = 'I would like to meet you again at the next Google conference in California.' \
       ' There would be about 1000 people and you should pay 100.00 $ on early booking and 500.00 after.' \
       ' See you on 10/21/2019.'

doc = nlp(text)
spacy.tokens.token.Token.set_extension('transient', default='')