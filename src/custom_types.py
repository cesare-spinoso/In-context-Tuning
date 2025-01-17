from typing import NamedTuple, TypedDict


Task = str
"""Name of the task. For LAMA, tasks are given special codenames like P530."""

ModelInput = str
"""String that will be passed to the model. Consists of K examples with their
answers and a query example with its answer masked out. For LAMA, this looks
like: 'France has diplomatic relations with Germany. Canada works with the
U.S. The U.K. is allied with [MASK].'"""


class TrainingExample(NamedTuple):
    """A single training example that will be passed to the language model.
    Consists of the task name, the model input and the label."""

    task: Task
    model_input: ModelInput
    label: int


TrainingBatch = list[TrainingExample]
"""A single training batch should contain training examples all from the same task."""


class Example(TypedDict):
    """Single example for a task, consists of input and corresponding label.
    e.g. For LAMA, the input is the subject and the label is the object."""

    _input: str
    """The input to the task e.g. For LAMA, the name of a country."""
    _label: int
    """The label of the task for the given input.
	e.g. For LAMA, the name of a country that maintains diplomatic relations with the input country."""


class TemplateExample(TypedDict):
    """Similar to Example, but also includes the template associated with the datapoint.
    In this way, there is no multiple possible templates for each example."""

    _input: str
    """The input to the task e.g. For LAMA, the name of a country.
    For BiClfs, typically a sentence-like datapoint."""
    template: str
    """The template associated with this example's input and label.
    For e.g., <input> Does this review has a positive sentiment? <label>."""
    _label: int
    """The label of the example with the given input. e.g. For LAMA, the name
    of a country that is at war with the <input> country. For BiClfs, this is
    either a yes or no answer."""


Task2Examples = dict[Task, list[Example]]
"""List of all the examples (input, label pairs) for a given task.
e.g. For LAMA, the task might be the relations between two countries
and the examples might be pairs of countries."""

Task2TemplateExamples = dict[Task, list[Example]]

Template = str
"""Template for the task, used to fill in the input and label for a given example.
e.g. For LAMA, a template might be <input> maintains diplomatic relations with <label>."""

Task2Templates = dict[Task, list[Template]]
"""List of all the templates for a given task.
e.g. For LAMA, the task might be the diplomatic relations and the templates
might be <input> maintains diplomatic relations with <label> as well as other
variations."""

Verbalizer = str
"""There is a one-to-one mapping between verbalizers and class labels.
For LAMA, because each task is a 21K-way classification task, there are
21K verbalizers."""

ClassLabel = str
"""For LAMA, because each task is a 21K-way classification task, there are
21K class labels."""

Task2ClassLabels = dict[Task, list[ClassLabel]]
"""List of all the class labels for a given task.
e.g. For LAMA, each task will have the same list of class labels."""


class Prompt(TypedDict):
    prompt: str
    label: int


Task2Prompts = dict[Task, list[Prompt]]

# TODO
Task2Preds = None
Task2Scores = None
