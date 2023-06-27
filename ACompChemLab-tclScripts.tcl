###########################################################
## Copyright 2022 Atanu Acharya
## Thses scripts are developed by the ACompChemLab
## Research Lab of Dr. Atanu Acharya @ Syracuse University
## Contributor(s): Atanu Acharya
##  
##########################################################




##################################################
## Calculate number of atoms			##
## Usage: numatoms <molid> "<atom selection>"	##
## Default values:                      	##
## <molid> = top    <atom selection> = all	##
##################################################
proc numatoms {{mol top} {str "all"}} {
	set n_atoms [[atomselect $mol $str] num]
	return ${n_atoms}
} 
##################################################


##########################################################
## Create a dictionary of <atom name> <point charge>	##
## Usage: dictNameCharge <molid> "<atom selection>"	##
## Default values:                               	##
## <molid> = top    <atom selection> = all	        ##
##########################################################
proc dictNameCharge {{mol top} {str "all"}} {
	set nameChdict [dict create]
        set i_atoms [lsort -integer [[atomselect $mol "$str"] get index]]
	foreach i $i_atoms {
        	set name [[atomselect $mol "index $i"] get name]
        	set charge [format "%.4f" [[atomselect $mol "index $i"] get charge]]
		dict append nameChdict $name $charge
	}
	unset charge
	unset name
	return ${nameChdict} 
} 
###########################################################


##########################################################
## Create a dictionary of <atom index> <point charge>	##
## Usage: dictIndexCharge <molid> "<atom selection>"	##
## Default values:                               	##
## <molid> = top    <atom selection> = all	        ##
##########################################################
proc dictIndexCharge {{mol top} {str "all"}} {
	set indexChdict [dict create]
        set i_atoms [lsort -integer [[atomselect $mol "$str"] get index]]
	foreach i $i_atoms {
        	set charge [format "%.4f" [[atomselect $mol "index $i"] get charge]]
		dict append indexChdict $i $charge
	}
	unset charge 
	return ${indexChdict} 
} 
##########################################################

###################################################################
## Create a dictionary of <atom index> <Coordinate>              ##
## Usage: dictIndexCoor <molid> "<atom selection>"  <frame >     ##
## Default values:                                               ##
## <molid> = top    <atom selection> = all  <frame> = last       ##
###################################################################
proc dictIndexCoor {{mol top} {str "all"} {frame last}} {
        set indexCoordict [dict create]
        set i_atoms [lsort -integer [[atomselect $mol "$str"] get index]]
	foreach i $i_atoms {
                set coor [[atomselect $mol "index $i" frame $frame] get {x y z}]
                dict append indexCoordict $i $coor
        }
        unset coor
        return ${indexCoordict}
}
###################################################################

##########################################################################################
## Create a dictionary of all atom properties in the following format  			##
## <index> {name <name> charge <charge> xcoor <xcoor> ycoor <ycoor> zcoor <zcoor>}      ##
## Usage: parseAtoms <molid> <frame> "<atom selection>"         			##
## Default values:                                               			##
## <molid> = top    <frame> = last     <atom selection> = all      			##
##########################################################################################
proc parseAtoms {{mol top} {whichframe last} {str "all"}} {
        set i_atoms [lsort -integer [[atomselect $mol "$str"] get index]]
        set AllAtoms [dict create]

	foreach i $i_atoms {
                dict set AllAtoms $i name [[atomselect $mol "index $i"] get name]
                dict set AllAtoms $i charge [format "%.4f" [[atomselect $mol "index $i"] get charge]]
                dict set AllAtoms $i xcoor [format "%.4f" [[atomselect $mol "index $i" frame $whichframe] get x ]]
                dict set AllAtoms $i ycoor [format "%.4f" [[atomselect $mol "index $i" frame $whichframe] get y ]]
                dict set AllAtoms $i zcoor [format "%.4f" [[atomselect $mol "index $i" frame $whichframe] get z ]]
        }
        return ${AllAtoms}
}
##########################################################################################







