#------------------------------------------------------------
# Table: VENTES
#------------------------------------------------------------

CREATE TABLE IF NOT EXISTS VENTES(
        ID_VENTE        Int  Auto_increment  NOT NULL ,
        MONTANT         Int NOT NULL ,
        NUMERO_RUE      Varchar (50) NOT NULL ,
        RUE             Varchar (50) NOT NULL ,
        CODE_POSTAL     Varchar (50) NOT NULL ,
        LONGITUDE       Decimal (9,6) NOT NULL ,
        LATITUDE        Decimal (9,6) NOT NULL ,
        DATE_MUTATION   Datetime NOT NULL ,
        SURFACE_BATI    Int NOT NULL ,
        NB_PIECES       Int NOT NULL ,
        SURFACE_TERRAIN Int NOT NULL ,
        DEPENDANCES     Varchar (50) NOT NULL ,
        ID_TYPE_BIEN    Int NOT NULL ,
        ID_COMMUNE      Varchar (50) NOT NULL,
        PRIMARY KEY (ID_VENTE)
)ENGINE=InnoDB;
