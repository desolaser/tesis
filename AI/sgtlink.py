#!/usr/bin/python
from sigopt import Connection
from slackclient import SlackClient
import subprocess
import click

@click.command()
@click.option('--iterations', default=0, type=int, 
			 help='Quantity of evaluation iterations (if default then iterations equals expermient´s observation budget)')
@click.option('--delete_obs', is_flag=True, 
			 help='Delete previous observations')
def eval(iterations):
	"""Makes an eval loop between sigopt and the CNN code."""
	conn = Connection(client_token="REKZSVNIITYEDUTODXTSUXEJRACOBDIOKLAPBNRCGZXWQQAC")
	experiment = conn.experiments(52578)
	experiment.suggestions().delete()

	if iterations == 0:
		iterations = experiment.observation_budget

	if delete_obs:    	
		experiment.observations().delete()

	for _ in range(iterations):
		suggestion = experiment.suggestions().create()
		print("Iteración", _ + 1)
		print("Learning rate:", suggestion.assignments['alpha'])
		value = evaluate_model(suggestion.assignments)
		experiment.observations().create(
			suggestion=suggestion.id,
			value=value,
		)

	msg = "La recompensa final es igual a: {:.2f}".format(reward)
	print(msg)
	slack_push(msg)

def slack_push(msg):
	"""Sends a message to thesis slack channel.

	Keyword arguments:
	msg -- String that defines the message
	"""
	token = 'xoxp-422129945285-421269443440-432589457383-5239bf7d291ed71c82470912ed0291de'
	sc = SlackClient(token)
	sc.api_call('chat.postMessage', channel='tesis',
				text=msg, username='Programa Python',
				icon_emoji=':robot_face:')

def evaluate_model(assignments):
	"""Evaluates the model and returns the reward.

	Keyword arguments:
	assignments -- Suggested params from the sigopt system
	"""
	learning_rate = assignments['alpha']
	command = ('AI train --learning_rate={:.4f}'.format(learning_rate))

	p = subprocess.Popen(command, universal_newlines=True, shell=True, 
						 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	text = p.stdout.read()
	retcode = p.wait()

	reward = float(text.splitlines()[-1])
	msg = "La recompensa es igual a: {:.2f}".format(reward)
	print(msg)
	slack_push(msg)
	return reward

