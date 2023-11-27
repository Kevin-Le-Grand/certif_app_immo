#------------------------------------------------------------
#        Script MySQL.
#------------------------------------------------------------


#------------------------------------------------------------
# Table: REGIONS
#------------------------------------------------------------

CREATE TABLE IF NOT EXISTS REGIONS(
        ID_REGION   Int  Auto_increment  NOT NULL ,
        Name_region Varchar (50) NOT NULL
	,CONSTRAINT REGIONS_PK PRIMARY KEY (ID_REGION)
)ENGINE=InnoDB;


#------------------------------------------------------------
# Table: DEPARTEMENTS
#------------------------------------------------------------

CREATE TABLE IF NOT EXISTS DEPARTEMENTS(
        ID_DEPT          Varchar (3) NOT NULL ,
        Name_departement Varchar (50) NOT NULL ,
        ID_REGION        Int NOT NULL
	,CONSTRAINT DEPARTEMENTS_PK PRIMARY KEY (ID_DEPT)

	,CONSTRAINT DEPARTEMENTS_REGIONS_FK FOREIGN KEY (ID_REGION) REFERENCES REGIONS(ID_REGION)
)ENGINE=InnoDB;


#------------------------------------------------------------
# Table: TYPES_BIENS
#------------------------------------------------------------

CREATE TABLE IF NOT EXISTS TYPES_BIENS(
        ID_TYPE_BIEN   Int  Auto_increment  NOT NULL ,
        NAME_TYPE_BIEN Varchar (50) NOT NULL
	,CONSTRAINT TYPES_BIENS_PK PRIMARY KEY (ID_TYPE_BIEN)
)ENGINE=InnoDB;


#------------------------------------------------------------
# Table: COMMUNES
#------------------------------------------------------------

CREATE TABLE IF NOT EXISTS COMMUNES(
        ID_COMMUNE   Varchar (50) NOT NULL ,
        NAME_COMMUNE Varchar (50) NOT NULL ,
        ID_DEPT      Varchar (3) NOT NULL
	,CONSTRAINT COMMUNES_PK PRIMARY KEY (ID_COMMUNE)

	,CONSTRAINT COMMUNES_DEPARTEMENTS_FK FOREIGN KEY (ID_DEPT) REFERENCES DEPARTEMENTS(ID_DEPT)
)ENGINE=InnoDB;


#------------------------------------------------------------
# Table: VENTES
#------------------------------------------------------------

CREATE TABLE IF NOT EXISTS VENTES(
        ID_VENTE        Int  Auto_increment  NOT NULL ,
        MONTANT         Int NOT NULL ,
        NUMERO_RUE      Varchar (50) NOT NULL ,
        RUE             Varchar (50) NOT NULL ,
        CODE_POSTAL     Int NOT NULL ,
        LONGITUDE       Decimal (9,6) NOT NULL ,
        LATITUDE        Decimal (9,6) NOT NULL ,
        DATE            Datetime NOT NULL ,
        SURFACE_BATI    Int NOT NULL ,
        NB_PIECES       Int NOT NULL ,
        SURFACE_TERRAIN Int NOT NULL ,
        ID_TYPE_BIEN    Int NOT NULL ,
        ID_COMMUNE      Varchar (50) NOT NULL
	,CONSTRAINT VENTES_PK PRIMARY KEY (ID_VENTE)

	,CONSTRAINT VENTES_TYPES_BIENS_FK FOREIGN KEY (ID_TYPE_BIEN) REFERENCES TYPES_BIENS(ID_TYPE_BIEN)
	,CONSTRAINT VENTES_COMMUNES0_FK FOREIGN KEY (ID_COMMUNE) REFERENCES COMMUNES(ID_COMMUNE)
)ENGINE=InnoDB;

