import theme from 'compDir/Theme/capstone';

export default {
	tableContainer: {
		border: '1px solid rgba(0, 0, 0, 0.23)',
		borderRadius: theme.spacing(0.5),
		['& > .MuiTable-root']: {
			['& > .MuiTableHead-root']: {
				['& > .MuiTableRow-root']: {
					['& > th']: {
						fontSize: '11px !important',
						fontWeight: 700,
						padding: theme.spacing(1),
						// borderBottomColor: 'rgba(0, 0, 0, 0.23)',
					}
				}
			},
			['& > .MuiTableBody-root']: {
				['& > .MuiTableRow-root']: {
					['& > .MuiTableCell-root']: {
						padding: theme.spacing(1),
						fontSize: theme.spacing(1.5),
						maxWidth: theme.spacing(17),
						wordBreak: 'break-word',
						// borderBottomColor: 'rgba(0, 0, 0, 0.23)',
						fontWeight: 400,
						// ['& > button']: {
						// 	padding: theme.spacing(0.5, 1)
						// },
						['& > .MuiChip-root']: {
							minWidth: theme.spacing(7),
						}
					},
					['&:hover']: {
						background: 'rgba(0,0,0,0.08)'
					}
				}
			}
		}
	},
	rowBg: {
		background: '#FFFFFF'
	},
	row: {
		'&.active': {
			background: '#F57F1714 !important',
		},
		// ['&:nth-of-type(even) > td']: {
		// 	padding: '4px !important',
		// },
		'& > td.close': {
			paddingBlock: '0 !important',
			transitionDuration: '302ms'
		}
	} 
};