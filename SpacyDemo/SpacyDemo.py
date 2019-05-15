#from __future__ import unicode_literals, print_function

#import plac
#import random
#from pathlib import Path
#import spacy
#from spacy.util import minibatch, compounding


## new entity label
#LABEL = "Definition"
#LABEL_ITEM = "ITEM"
## training data
## Note: If you're using an existing model, make sure to mix in examples of
## other entity types that spaCy correctly recognized before. Otherwise, your
## model might learn the new type, but "forget" what it previously knew.
## https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

##TRAIN_DATA = [
##    (
##        "Horses are too tall and they pretend to care about your feelings",
##        {"entities": [(0, 6, LABEL)]},
##    ),
##    ("Do they bite?", {"entities": []}),
##    (
##        "horses are too tall and they pretend to care about your feelings",
##        {"entities": [(0, 6, LABEL)]},
##    ),
##    ("horses pretend to care about your feelings", {"entities": [(0, 6, LABEL)]}),
##    (
##        "they pretend to care about your feelings, those horses",
##        {"entities": [(48, 54, LABEL)]},
##    ),
##    ("horses?", {"entities": [(0, 6, LABEL)]}),
##]
#TRAIN_DATA = [
#    (
#        "A computer is a machine that can be instructed to carry out sequences of arithmetic or logical operations automatically via computer programming. Modern computers have the ability to follow generalized sets of operations, called programs. These programs enable computers to perform an extremely wide range of tasks. A 'complete' computer including the hardware, the operating system (main software), and peripheral equipment required and used for 'full' operation can be referred to as a computer system. This term may as well be used for a group of computers that are connected and work together, in particular a computer network or computer cluster.",
#        {"entities":[(0,145,LABEL),
#                     (2,10,LABEL_ITEM)]},
#        ),
#    ("In computer science, random-access machine (RAM) is an abstract machine in the general class of register machines. The RAM is very similar to the counter machine but with the added capability of 'indirect addressing' of its registers. Like the counter machine the RAM has its instructions in the finite-state portion of the machine (the so-called Harvard architecture).The RAM's equivalent of the universal Turing machine – with its program in the registers as well as its data – is called the random-access stored-program machine or RASP. It is an example of the so-called von Neumann architecture and is closest to the common notion of computer.",
#     {"entities":[(0,114,LABEL),
#                  (44,47,LABEL_ITEM)]}
#     ),
#     ("Vocabulary is the collection of words that an individual knows (Linse,2005:121). There are some experts who give definitions of vocabulary. Hatch and Brown (1995:1) define that vocabulary as a list of words for a particular language or a list or set of word that individual speakers of language might use.",
#      {"entities":[(0,80,LABEL),
#                   (0,10,LABEL_ITEM)]}),
#     ("Collecting training data may sound incredibly painful – and it can be, if you’re planning a large-scale annotation project. However, if your main goal is to update an existing model’s predictions – for example, spaCy’s named entity recognition – the hard part is usually not creating the actual annotations. It’s finding representative examples and extracting potential candidates. The good news is, if you’ve been noticing bad performance on your data, you likely already have some relevant text, and you can use spaCy to bootstrap a first set of training examples. For example, after processing a few sentences, you may end up with the following entities, some correct, some incorrect.",
#        {"entities":[(211,307,LABEL),
#                        (211,243,LABEL_ITEM)]}),
#                        ("Smartphones are a class of mobile phones and of multi-purpose mobile computing devices. They are distinguished from feature phones by their stronger hardware capabilities and extensive mobile operating systems, which facilitate wider software, internet (including web browsing[1] over mobile broadband), and multimedia functionality (including music, video, cameras, and gaming), alongside core phone functions such as voice calls and text messaging. Smartphones typically include various sensors that can be leveraged by their software, such as a magnetometer, proximity sensors, barometer, gyroscope and accelerometer, and support wireless communications protocols such as Bluetooth, Wi-Fi, and satellite navigation.",
#                        {"entities":[(0,87,LABEL),
#                                    (0,11,LABEL_ITEM)]},),
#                                      ("A car is a wheeled motor vehicle used for transporting passengers.",
#     {"entities": [(0, 66, LABEL),
#                   (2, 5, LABEL_ITEM)]},
#     ),
#    ("The United Nations (UN) is an intergovernmental organization that was tasked to maintain international peace and security, develop friendly relations among nations, achieve international co-operation and be a centre for harmonizing the actions of nations. The headquarters of the UN is in Manhattan, New York City, and is subject to extraterritoriality. Further main offices are situated in Geneva, Nairobi, Vienna and The Hague. The organization is financed by assessed and voluntary contributions from its member states. Its objectives include maintaining international peace and security, protecting human rights, delivering humanitarian aid, promoting sustainable development and upholding international law.",
#     {"entities": [(0, 255, LABEL),
#                   (4, 18, LABEL_ITEM)]},
#     ),
#    ("An atom is the smallest constituent unit of ordinary matter that has the properties of a chemical element.",
#     {"entities": [(0, 107, LABEL),
#                   (3, 7, LABEL_ITEM)]},
#     ),
#    ("A country is a region that is identified as a distinct entity in political geography.",
#     {"entities": [(0, 85, LABEL),
#                   (2, 9, LABEL_ITEM)]},
#     )
#    ]

#@plac.annotations(
#    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
#    new_model_name=("New model name for model meta.", "option", "nm", str),
#    output_dir=("Optional output directory", "option", "o", Path),
#    n_iter=("Number of training iterations", "option", "n", int),
#)
#def main(model=None, new_model_name="animal", output_dir=None, n_iter=1000):
#    """Set up the pipeline and entity recognizer, and train the new entity."""
#    random.seed(0)
#    if model is not None:
#        nlp = spacy.load(model)  # load existing spaCy model
#        print("Loaded model '%s'" % model)
#    else:
#        nlp = spacy.blank("en")  # create blank Language class
#        print("Created blank 'en' model")
#    nlp = spacy.load("en_core_web_sm")
#    # Add entity recognizer to model if it's not in the pipeline
#    # nlp.create_pipe works for built-ins that are registered with spaCy
#    if "ner" not in nlp.pipe_names:
#        ner = nlp.create_pipe("ner")
#        nlp.add_pipe(ner)
#    # otherwise, get it, so we can add labels to it
#    else:
#        ner = nlp.get_pipe("ner")

#    ner.add_label(LABEL)  # add new entity label to entity recognizer
#    ner.add_label(LABEL_ITEM)  # add new entity label to entity recognizer
#    # Adding extraneous labels shouldn't mess anything up
#    ner.add_label("VEGETABLE")
#    if model is None:
#        optimizer = nlp.begin_training()
#    else:
#        optimizer = nlp.resume_training()
#    move_names = list(ner.move_names)
#    # get names of other pipes to disable them during training
#    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
#    with nlp.disable_pipes(*other_pipes):  # only train NER
#        sizes = compounding(1.0, 4.0, 1.001)
#        # batch up the examples using spaCy's minibatch
#        for itn in range(n_iter):
#            random.shuffle(TRAIN_DATA)
#            batches = minibatch(TRAIN_DATA, size=sizes)
#            losses = {}
#            for batch in batches:
#                texts, annotations = zip(*batch)
#                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
#            print("Losses", losses)

#    # test the trained model
#    test_text = "House is a flat. Fucking shit."
#    doc = nlp(test_text)
#    print("Entities in '%s'" % test_text)
#    for ent in doc.ents:
#        print(ent.label_, ent.text)

#    # save model to output directory
#    if output_dir is not None:
#        output_dir = Path(output_dir)
#        if not output_dir.exists():
#            output_dir.mkdir()
#        nlp.meta["name"] = new_model_name  # rename model
#        nlp.to_disk(output_dir)
#        print("Saved model to", output_dir)

#        # test the saved model
#        print("Loading from", output_dir)
#        nlp2 = spacy.load(output_dir)
#        # Check the classes have loaded back consistently
#        assert nlp2.get_pipe("ner").move_names == move_names
#        doc2 = nlp2(test_text)
#        for ent in doc2.ents:
#            print(ent.label_, ent.text)


#if __name__ == "__main__":
#    plac.call(main)

from spacy.matcher import Matcher
import spacy
def on_match(matcher, doc, id, matches):
    print('Matched!', matcher, doc, id, matches)
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
matcher.add("HelloWorld", on_match, [{"LEMMA": {"IN": ["like", "love"]}},
            {"POS": "VERB"}])
matcher.add("Definition",on_match,[{"POS":"DET","OP":"?"},
 {"POS":"NOUN"},
 {"LEMMA":{"IN": ["be","-"]}, },
 {"POS":"DET","OP":"?"},
 {"POS":"VERB"}])
doc = nlp(u"HTTP is completed and i will do next task")
matches = matcher(doc)  