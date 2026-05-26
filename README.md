# LLM Red Teaming for RAG Systems

## Problem Statement

RAG (Retrieval-Augmented Generation) pipelines are increasingly used in production 
LLM applications, but their security properties are poorly understood. Unlike 
standard LLM deployments, RAG systems introduce an additional attack surface — 
the retrieval component — where adversarial inputs can influence what context 
gets injected into the model prompt.

This project investigates vulnerabilities in RAG-augmented LLM pipelines through 
automated red teaming. The goal is to systematically find failure modes across 
both the retrieval and generation stages, and use those findings to inform 
practical defenses.

## What This Project Does

- Builds a target RAG pipeline (FAISS vector store + LLM backend) as the system 
  under test
- Designs an adversarial attacker agent that generates and refines attack strategies
- Covers attack categories including prompt injection, indirect injection via 
  retrieved documents, goal hijacking, and role confusion
- Evaluates attacks using metrics: attack success rate, refusal bypass rate, 
  response harmfulness

## Status

Work in progress. RAG pipeline is complete. Attacker agent under active development.

## Tech Stack

Python, LangChain, FAISS, Transformers
