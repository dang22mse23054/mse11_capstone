import { Box, BoxProps, Grid } from '@material-ui/core';
import React, { FC, ReactNode, Fragment } from 'react';
import MDIcon from '@mdi/react';

interface IProps extends BoxProps {
	icon?: any
	iconSize?: any
	isRequired?: boolean
	value: string | ReactNode
	spacing?: number
}

const CustomLabel: FC<IProps> = (props: IProps) => {
	const { icon, iconSize, isRequired, value, spacing = 1, ...boxProps } = props;

	const label = () => typeof value == 'string' ? (
		<Box overflow='hidden' textOverflow="ellipsis" {...boxProps} >
			{value}
		</Box>
	) : value;

	return (
		<Grid container alignItems='center'>
			{icon && (
				<Fragment>
					<MDIcon size={iconSize || '13pt'} path={icon} style={{ verticalAlign: 'middle' }} />
					{
						Array(spacing).fill(0).map((val, i) => <Fragment key={i}>&nbsp;</Fragment>)
					}
				</Fragment>
			)}
			{label()}
			{isRequired && <Box fontSize={15} style={{ color: 'red' }}>&nbsp;※</Box>}
		</Grid>
	);
};


export default CustomLabel;