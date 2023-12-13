(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["i18n-demo"],{"4c91":function(e,t,a){"use strict";a.r(t);var l=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",[a("el-card",{staticClass:"box-card",staticStyle:{"margin-top":"40px"}},[a("div",{staticClass:"clearfix",attrs:{slot:"header"},slot:"header"},[a("svg-icon",{attrs:{name:"international"}}),a("span",{staticStyle:{"margin-left":"10px"}},[e._v(e._s(e.$t("i18nView.title")))])],1),a("div",[a("el-radio-group",{attrs:{size:"small"},model:{value:e.lang,callback:function(t){e.lang=t},expression:"lang"}},[a("el-radio",{attrs:{label:"zh",border:""}},[e._v(" 简体中文 ")]),a("el-radio",{attrs:{label:"en",border:""}},[e._v(" English ")]),a("el-radio",{attrs:{label:"es",border:""}},[e._v(" Español ")]),a("el-radio",{attrs:{label:"ja",border:""}},[e._v(" 日本語 ")]),a("el-radio",{attrs:{label:"ko",border:""}},[e._v(" 한국어 ")]),a("el-radio",{staticStyle:{"margin-left":"0","margin-top":"10px"},attrs:{label:"it",border:""}},[e._v(" Italiano ")])],1),a("el-tag",{staticStyle:{"margin-top":"15px",display:"block"},attrs:{type:"info"}},[e._v(" "+e._s(e.$t("i18nView.note"))+" ")])],1)]),a("el-row",{staticStyle:{margin:"100px 15px 50px"},attrs:{gutter:20}},[a("el-col",{attrs:{span:12,xs:24}},[a("div",{staticClass:"block"},[a("el-date-picker",{attrs:{placeholder:e.$t("i18nView.datePlaceholder"),type:"date"},model:{value:e.date,callback:function(t){e.date=t},expression:"date"}})],1),a("div",{staticClass:"block"},[a("el-select",{attrs:{placeholder:e.$t("i18nView.selectPlaceholder")},model:{value:e.value,callback:function(t){e.value=t},expression:"value"}},e._l(e.options,(function(e){return a("el-option",{key:e.value,attrs:{label:e.label,value:e.value}})})),1)],1),a("div",{staticClass:"block"},[a("el-button",{staticClass:"item-btn",attrs:{size:"small"}},[e._v(" "+e._s(e.$t("i18nView.default"))+" ")]),a("el-button",{staticClass:"item-btn",attrs:{size:"small",type:"primary"}},[e._v(" "+e._s(e.$t("i18nView.primary"))+" ")]),a("el-button",{staticClass:"item-btn",attrs:{size:"small",type:"success"}},[e._v(" "+e._s(e.$t("i18nView.success"))+" ")]),a("el-button",{staticClass:"item-btn",attrs:{size:"small",type:"info"}},[e._v(" "+e._s(e.$t("i18nView.info"))+" ")]),a("el-button",{staticClass:"item-btn",attrs:{size:"small",type:"warning"}},[e._v(" "+e._s(e.$t("i18nView.warning"))+" ")]),a("el-button",{staticClass:"item-btn",attrs:{size:"small",type:"danger"}},[e._v(" "+e._s(e.$t("i18nView.danger"))+" ")])],1)]),a("el-col",{attrs:{span:12,xs:24}},[a("el-table",{staticStyle:{width:"100%"},attrs:{data:e.tableData,fit:"","highlight-current-row":"",border:""}},[a("el-table-column",{attrs:{label:e.$t("i18nView.tableName"),prop:"name",width:"100",align:"center"}}),a("el-table-column",{attrs:{label:e.$t("i18nView.tableDate"),prop:"date",width:"120",align:"center"}}),a("el-table-column",{attrs:{label:e.$t("i18nView.tableAddress"),prop:"address"}})],1)],1)],1)],1)},i=[],s=a("d4ec"),n=a("bee2"),r=a("262e"),o=a("2caf"),c=a("9ab4"),d=a("1b40"),b=a("ac1a"),u={zh:{i18nView:{title:"切换语言",note:"本项目国际化基于 vue-i18n",datePlaceholder:"请选择日期",selectPlaceholder:"请选择",tableDate:"日期",tableName:"姓名",tableAddress:"地址",default:"默认按钮",primary:"主要按钮",success:"成功按钮",info:"信息按钮",warning:"警告按钮",danger:"危险按钮",one:"一",two:"二",three:"三"}},en:{i18nView:{title:"Switch Language",note:"The internationalization of this project is based on vue-i18n",datePlaceholder:"Pick a day",selectPlaceholder:"Select",tableDate:"tableDate",tableName:"tableName",tableAddress:"tableAddress",default:"default",primary:"primary",success:"success",info:"info",warning:"warning",danger:"danger",one:"One",two:"Two",three:"Three"}},es:{i18nView:{title:"Cambiar idioma",note:"La internacionalización de este proyecto se basa en vue-i18n",datePlaceholder:"Escoge un día",selectPlaceholder:"Seleccionar",tableDate:"tableDate",tableName:"tableName",tableAddress:"tableAddress",default:"defecto",primary:"primario",success:"éxito",info:"info",warning:"advertencia",danger:"peligro",one:"Uno",two:"Dos",three:"Tres"}},ja:{i18nView:{title:"言語切替",note:"vue-i18nを使っています",datePlaceholder:"日時選択",selectPlaceholder:"選択してください",tableDate:"日時",tableName:"姓名",tableAddress:"住所",default:"default",primary:"primary",success:"success",info:"info",warning:"warning",danger:"danger",one:"1",two:"2",three:"3"}},ko:{i18nView:{title:"언어 변경",note:"이 프로젝트의 국제화는 vue-i18n을 기반으로합니다",datePlaceholder:"요일 선택",selectPlaceholder:"선택",tableDate:"테이블 날짜",tableName:"테이블 이름",tableAddress:"테이블 주소",default:"고정 값",primary:"1순위",success:"성공",info:"정보",warning:"경고",danger:"위험",one:"하나",two:"둘",three:"셋"}},it:{i18nView:{title:"Cambia Lingua",note:"L'internalizzazione di questo progetto è basata su vue-i18n",datePlaceholder:"Scegli un giorno",selectPlaceholder:"Seleziona",tableDate:"Data",tableName:"Nome",tableAddress:"Indirizzo",default:"default",primary:"primario",success:"successo",info:"info",warning:"attenzione",danger:"pericolo",one:"Uno",two:"Due",three:"Tre"}}},g=function(e){Object(r["a"])(a,e);var t=Object(o["a"])(a);function a(){var e;return Object(s["a"])(this,a),e=t.apply(this,arguments),e.date="",e.value="",e.options=[],e.tableData=[{date:"2016-05-03",name:"Tom",address:"No. 189, Grove St, Los Angeles"},{date:"2016-05-02",name:"Tom",address:"No. 189, Grove St, Los Angeles"},{date:"2016-05-04",name:"Tom",address:"No. 189, Grove St, Los Angeles"},{date:"2016-05-01",name:"Tom",address:"No. 189, Grove St, Los Angeles"}],e}return Object(n["a"])(a,[{key:"created",value:function(){var e="i18nView";this.$i18n.getLocaleMessage("en")[e]||(this.$i18n.mergeLocaleMessage("en",u.en),this.$i18n.mergeLocaleMessage("zh",u.zh),this.$i18n.mergeLocaleMessage("es",u.es),this.$i18n.mergeLocaleMessage("ja",u.ja),this.$i18n.mergeLocaleMessage("ko",u.ko),this.$i18n.mergeLocaleMessage("it",u.it)),this.setOptions()}},{key:"setOptions",value:function(){this.options=[{value:"1",label:this.$t("i18nView.one")},{value:"2",label:this.$t("i18nView.two")},{value:"3",label:this.$t("i18nView.three")}]}},{key:"lang",get:function(){return b["a"].language},set:function(e){b["a"].SetLanguage(e),this.$i18n.locale=e,this.setOptions()}}]),a}(d["c"]);g=Object(c["a"])([Object(d["a"])({name:"I18n"})],g);var m=g,p=m,h=(a("83fe"),a("2877")),v=Object(h["a"])(p,l,i,!1,null,"551627d9",null);t["default"]=v.exports},"83fe":function(e,t,a){"use strict";a("e19f")},e19f:function(e,t,a){}}]);