# Changelog

All notable changes to ezQMMM will be documented in this file.

## [2.0.1] — 2026-04-14

### Fixed
CS charge position bug: It will now add 6% of bond length by defaults

Printing bug: Missing virtual charge printing for CS

Bug: Discripency between QM/MM input and PDB writer

### Added
Charge suggestions.
 
Warning for large differences between QM charge and MM charge of the QM region as obtained from the topology.

QM-MM cuts mentioned while executing.

Warning for polar bond cuts

Added estimated time and summary block

User may provide CS scaling for location of dummy charges.

User may provide the wrapping strategy. Default is residue

### Changed
Example yaml file uses mm_switchdist value by default 

Only one mapping function with PBC-aware remapping of atoms/residues/fragments around QM

